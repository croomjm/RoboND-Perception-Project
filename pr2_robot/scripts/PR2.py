#!/usr/bin/env python
import pickle
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *
from math import pi

import rospy
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

from segmenter import Segmenter

class dropbox_data(object):
    def __init__(self, position, arm):
        self.pos = position
        self.arm = arm
    def show():
        print("arm = {0:s}, pos = {1:f}".format(self.arm, self.pos))

class pick_place_data(object):
    def __init__(self):
        self.test_scene_num = Int32()
        self.arm_name = String()
        self.object_name = String()
        self.object_group = String()
        self.pick_pose = Pose()
        self.place_pose = Pose()
        self.pick_pose_point = Point()
        self.place_pose_point = Point()

    def return_yaml_dict(self):
        # Return a yaml friendly dictionary of ppd attributes
        yaml_dict = {}
        yaml_dict["test_scene_num"] = self.test_scene_num.data
        yaml_dict["arm_name"]  = self.arm_name.data
        yaml_dict["object_name"] = self.object_name.data
        yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(self.pick_pose)
        yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(self.place_pose)
        return yaml_dict

class PR2(object):
    def __init__(self, model_file):
        self.max_success_count = 0
        self.success_count = 0
        self.dropbox_dict = {}
        self.dict_list = []
        self.object_list = None
        self.color_list = []

        self.table_cloud = None
        self.detected_objects = None

        #load SVM object classifier
        self.model = pickle.load(open(model_file, 'rb'))

        #initialize object segmenter
        self.segmenter = Segmenter(self.model)

        #initialize point cloud publishers
        self.objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
        self.table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
        self.colorized_cluster_pub = rospy.Publisher("/colorized_clusters", PointCloud2, queue_size=1)
        self.object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
        self.detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size = 1)
        #self.denoised_pub = rospy.Publisher("/denoised_cloud", PointCloud2, queue_size = 1)
        self.reduced_cloud_pub = rospy.Publisher("/decimated_cloud", PointCloud2, queue_size = 1)

        self.collision_cloud_pub = rospy.Publisher("/pr2/3d_map/points", PointCloud2, queue_size = 1)

        print('PR2 object initialized.')

    ##YAML Utilities##

    def send_to_yaml(self, yaml_filename, dict_list):
        # Helper function to output to yaml file
        data_dict = {"object_list": dict_list}
        with open(yaml_filename, 'w') as outfile:
            yaml.dump(data_dict, outfile, default_flow_style=False)

    ##PR2 mover##
    def joint_state_handler(self,msg):
        print('Joint state received.')
        index = msg.name.index('world_joint')
        position = msg.position[index]
        print('World Joint position: {}'.format(position))

    def segment_scene(self, pcl_msg):
        seg = self.segmenter #to reduce verbosity below

        # TODO: Convert ROS msg to PCL data
        cloud = ros_to_pcl(pcl_msg)
        leaf_size = 0.005

        # TODO: Voxel Grid Downsampling
        print('Reducing voxel resolution.')
        cloud = seg.voxel_grid_downsample(cloud, leaf_size = leaf_size)
        decimated_cloud = cloud

        #Reduce outlier noise in object cloud
        print('Rejecting outliers in raw cloud.')
        cloud = seg.outlier_filter(cloud, 15, 0.01)
        denoised_cloud = cloud

        # TODO: PassThrough Filter
        print('Applying passthrough filters.')
        cloud = seg.axis_passthrough_filter(cloud, 'z', (0.55, 2)) #filter below table
        #passthroughz_cloud = cloud
        cloud = seg.axis_passthrough_filter(cloud, 'x', (.35, 10)) #filter out table front edge
        #passthroughy_cloud = cloud

        # TODO: RANSAC Plane Segmentation
        # TODO: Extract inliers and outliers
        print('Performing plane segmentation.')
        table_cloud, objects_cloud = seg.ransac_plane_segmentation(cloud, max_distance = leaf_size)

        #Reduce outlier noise in object cloud
        print('Rejecting outliers in objects cloud.')
        objects_cloud = seg.outlier_filter(objects_cloud, 10, 0.01)

        # TODO: Euclidean Clustering
        # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
        print('Finding object clusters.')
        cluster_indices = seg.get_euclidean_cluster_indices(objects_cloud, 0.03, (10,5000))
        colorized_object_clusters = seg.return_colorized_clusters(objects_cloud, cluster_indices, self.color_list)
        detected_objects, detected_objects_dict = seg.detect_objects(objects_cloud, cluster_indices)



        # TODO: Convert PCL data to ROS messages
        # TODO: Publish ROS messages
        print('Converting PCL data to ROS messages.')
        message_pairs = [(decimated_cloud, self.reduced_cloud_pub),
                         #(denoised_cloud, self.denoised_pub),
                         (objects_cloud, self.objects_pub),
                         (table_cloud, self.table_pub),
                         (colorized_object_clusters, self.colorized_cluster_pub)
                         #(passthroughy_cloud, self.passthroughy_filter_pub),
                         #(passthroughz_cloud, self.passthroughz_filter_pub),
                         ]
        
        seg.convert_and_publish(message_pairs)

        #publish detected objects and labels
        seg.publish_detected_objects(detected_objects,
                                     self.object_markers_pub,
                                     self.detected_objects_pub)

        self.object_list = detected_objects_dict
        self.detected_objects = detected_objects
        self.table_cloud = table_cloud

    def capture_collision_map(self):
        #publish joint positins from 0 to 2*pi to survey entire environment
        pub_j1 = rospy.Publisher('/pr2/world_joint_controller/command',
                                 Float64, queue_size=10)

        rate = rospy.Rate(10)
        increments = 30
        positions = range(increments)
        increments = increments*1.0

        for i in positions:
            position = i/increments*2*pi
            pub_j1.publish(position)
            rate.sleep()


        pub_j1.publish(0)

    def publish_collision_map(self,object_name,picked_objects):
        obstacle_cloud_list = list(self.table_cloud)

        for obj in self.detected_objects:
            if obj.label != object_name and obj.label not in picked_objects:
                obj_cloud = ros_to_pcl(obj.cloud)
                obstacle_cloud_list.extend(list(obj_cloud))
                print(obstacle_cloud)

        obstacle_cloud = pcl.PointCloud_PointXYZRGB()
        obstacle_cloud.from_list(obstacle_cloud_list)

        self.segmenter.convert_and_publish([(obstacle_cloud, )])

    def find_pick_object(self, obj):
        ppd = pick_place_data()

        group = obj['group']
        name = obj['name']

        #See if object is found within object list
        #if so, use position to populate pick_place_data object
        pos = self.object_list.get(name)

        if pos is not None:

            # TODO: Get the PointCloud for a given object and obtain it's centroid
            ppd.object_name.data = name
            ppd.pick_pose_point.x = np.asscalar(pos[0])
            ppd.pick_pose_point.y = np.asscalar(pos[1])
            ppd.pick_pose_point.z = np.asscalar(pos[2])
            ppd.pick_pose.position = pick_pose_point

            # TODO: Create 'place_pose' for the object
            dropboxdata = self.dropbox_dict[group]
            ppd.place_pose_point.x = dropboxdata.pos[0]
            ppd.place_pose_point.y = dropboxdata.pos[1]
            ppd.place_pose_point.z = dropboxdata.pos[2]
            ppd.place_pose.position = place_pose_point

            # TODO: Assign the arm to be used for pick_place
            ppd.arm_name.data = dropboxdata.arm

            print("Scene %d, picking up found %s object, using %s arm, and placing it in the %s bin." % 
                      (ppd.test_scene_num.data, ppd.object_name.data, ppd.arm_name.data, group))

            # TODO: Create a list of dictionaries (made with make_yaml_dict())
            # for later output to yaml format
            yaml_dict = self.make_yaml_dict(ppd)
            self.dict_list.append(yaml_dict)
            self.success_count += 1
            return ppd

        else:
            print("Label: %s not found" % name)
            return None

    def get_pick_object(self, ppd):
        # Use pick_place_routine service proxy to get the object
        print('Waiting for "pick_place_routine" service...')
        rospy.wait_for_service('pick_place_routine')

        try:
            resp = pick_place_routine(ppd.test_scene_num,
                                      ppd.object_name,
                                      ppd.arm_name,
                                      ppd.pick_pose,
                                      ppd.place_pose)

            print('Response: '.format(resp.success))

        except rospy.ServiceException as e:
            print('Service call failed: {}'.format(e))

    def mover(self):
        print('Starting mover method.')

        # TODO: Get/Read parameters
        object_list_param = rospy.get_param('/object_list')

        # Get scene number from launch file
        test_scene_num = rospy.get_param('/test_scene_num')
        print("test_scene_num = %d"% test_scene_num)

        # Get dropbox parameters
        dropbox_param = rospy.get_param('/dropbox')

        # TODO: Parse parameters into individual variables
        for dropbox in dropbox_param:
            dropboxdata = dropbox_data(dropbox['position'], dropbox['name'])
            self.dropbox_dict[dropbox['group']] = dropboxdata

        # TODO: Loop through the pick list
        picked = []
        self.publish_collision_map('', picked)
        # for obj in object_list_param:
        #     #get ppd object containing pick and place parameters
        #     ppd = self.find_pick_object(obj)

        #     #if successful, request that pick_place_routine service
        #     #publish all other objects + table to collision map
        #     #get the object
        #     if ppd is not None:
        #         self.publish_collision_map(obj, picked)
        #         self.get_pick_object(ppd)

        #         #update list of picked objects
        #         picked.append(obj['name'])

        # # TODO: Output your request parameters into output yaml file
        # if self.max_success_count < self.success_count:
        #     yaml_filename = "./output/output_" + str(test_scene_num.data) + ".yaml"
        #     print("output file name = %s" % yaml_filename)
        #     self.send_to_yaml(yaml_filename, self.dict_list)
        #     self.max_success_count = success_count

        # print("Success picking up object number = %d" % success_count)

    def pcl_callback(self, pcl_msg):
        # TODO: Rotate PR2 in place to capture side tables for the collision map
        #self.capture_collision_map()

        #segment scene and detect objects
        self.segment_scene(pcl_msg)

        #identify the objects listed in the pick list
        #submit them to the pick_place_routine
        self.mover()

def main():
    #model_file = '../training/model_sigmoid_300_orientations.sav'
    #model_file = '../training/model_sigmoid_50_orientations_YCbCr.sav'
    model_file = '../training/model_100_orientations_sigmoid_YCbCr_16_bin.sav'

    # TODO: ROS node initialization
    rospy.init_node('pr2', anonymous=True)

    pr2 = PR2(model_file)

    #pr2.capture_collision_map()

    pr2_pose_sub = rospy.Subscriber("joint_states", JointState, pr2.joint_state_handler, queue_size = 1)
    #initialize point cloud subscriber
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pr2.pcl_callback, queue_size=1)
    

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except:
        pass