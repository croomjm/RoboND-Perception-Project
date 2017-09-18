#!/usr/bin/env python

import numpy as np
import sklearn, pickle, pcl
from sklearn.preprocessing import LabelEncoder

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

class Segmenter(object):
    def __init__(self, model):
        #assign SVM model parameters
        self.clf = model['classifier']
        self.encoder = LabelEncoder()
        self.encoder.classes_ = model['classes']
        self.scaler = model['scaler']

    ##Publisher Methods##

    def convert_and_publish(self, message_pairs):
        #take in array of values and publisher channels
        #publish all values in array to corresponding publisher channels
        for m in message_pairs:
            cloud, publisher = m
            ros_cloud = pcl_to_ros(cloud)
            publisher.publish(ros_cloud)
            print('     Publishing {}.'.format(publisher.name))

    def publish_detected_objects(self, detected_objects, marker_pub, objects_pub):
        #publish all detected objects' labels to ros
        #publish the array of detected objects to ros

        for index, do in enumerate(detected_objects):
            # Publish the label into RViz
            centroid = self.get_centroid(ros_to_pcl(do.cloud))
            marker_loc = [centroid[0], centroid[1], centroid[2] + 0.2]
            marker_pub.publish(make_label(do.label,marker_loc,index))

        objects_pub.publish(detected_objects)

    ##Point Cloud Filters##

    def axis_passthrough_filter(self, cloud, axis, bounds):
        #apply a passthrough filter to a cloud
        #all points outside the bounds on the input axis (x,y, or z)
        #are removed from the returned cloud

        assert(axis in ['x','y','z'])

        passthrough = cloud.make_passthrough_filter()
        passthrough.set_filter_field_name(axis)
        axis_min, axis_max = bounds
        passthrough.set_filter_limits(axis_min, axis_max)
        cloud = passthrough.filter()

        return cloud    

    def outlier_filter(self, cloud, n_neighbors, threshhold_scale):
        #apply statistical outlier filter to cloud given
        #number of nearest neighbors to analyze for each point
        #and threshhold scale
        #return cloud with any point with mean neighbor distance
        #greater than global mean_neighbor_distance+threshhold_scale*std_dev

        filt = cloud.make_statistical_outlier_filter()
        filt.set_mean_k(n_neighbors)
        filt.set_std_dev_mul_thresh(threshhold_scale)
        cloud = filt.filter()

        return cloud

    def ransac_plane_segmentation(self, cloud, max_distance = 0.01):
        #take in cloud and the maximum distance from a segmented plane
        #return 2 clouds, one with inliers within the specified max distance,
        #and an outlier cloud with points outside the max distance
        seg = cloud.make_segmenter() 
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(max_distance)

        inliers, coefficients = seg.segment()

        outlier_cloud = cloud.extract(inliers, negative=True)
        inlier_cloud = cloud.extract(inliers, negative=False)

        return inlier_cloud, outlier_cloud

    def voxel_grid_downsample(self, cloud, leaf_size=0.01):
        #decimate cloud using a specified square voxel
        #side length
        vox = cloud.make_voxel_grid_filter()
        vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
        cloud = vox.filter()

        return cloud    

    ##Object Detection##

    def detect_objects(self, cloud, cluster_indices):
        #take in cloud and array of point indices within clusters
        #return array of detected objects within cloud along with
        #dict of the object labels and their corresponding centroids

        # Classify the clusters! (loop through each detected cluster one at a time)
        detected_objects_labels = []
        detected_objects = []
        #detected_objects_list = {'labels':[], 'centroids':[]}
        positions = []
        centroids = []

        for j, indices in enumerate(cluster_indices):
            # Grab the points for the cluster
            cluster = cloud.extract(indices)

            centroid = self.get_centroid(cluster)

            #set the starting position of the label
            #based on first point in cluster
            #offset later during label publishing
            positions.append(list(cluster[0][:3]))

            cluster = pcl_to_ros(cluster)

            # Compute the associated feature vector
            chists = compute_color_histograms(cluster)#, using_hsv=True)
            normals = self.get_normals(cluster)
            nhists = compute_normal_histograms(normals)
            feature_vector = np.concatenate((chists, nhists))

            # Make the prediction
            prediction = self.clf.predict(self.scaler.transform(feature_vector.reshape(1,-1)))
            label = self.encoder.inverse_transform(prediction)[0]

            # Add the detected object to the list of detected objects.
            do = DetectedObject()
            do.label = label
            do.cloud = cluster
            detected_objects.append(do)
            detected_objects_labels.append(label)
            centroids.append(centroid)

        detected_objects_dict = dict(zip(detected_objects_labels, centroids))

        rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

        return detected_objects, detected_objects_dict

    def get_euclidean_cluster_indices(self, cloud, tolerance, size_bounds):
        #return array of indices of clusters points spaced by a maximum
        #distance of tolerance and of total point cloud size within
        #size_bounds=(min num, max num)
        XYZcloud = XYZRGB_to_XYZ(cloud)

        min_cluster_size, max_cluster_size = size_bounds
        tree = XYZcloud.make_kdtree()
        
        # Create a cluster extraction object
        ec = XYZcloud.make_EuclideanClusterExtraction()
        
        # Set tolerances for distance threshold 
        # as well as minimum and maximum cluster size (in points)
        ec.set_ClusterTolerance(tolerance)
        ec.set_MinClusterSize(min_cluster_size)
        ec.set_MaxClusterSize(max_cluster_size)
        
        # Search the k-d tree for clusters
        ec.set_SearchMethod(tree)
        
        # Extract indices for each of the discovered clusters
        cluster_indices = ec.Extract()

        return cluster_indices

    ##Utility Methods##

    def get_centroid(self, cloud):
        #return centroid (x,y,z) of a pcl cloud
        points = np.mean(cloud.to_array(), axis=0)[:3]
        points = points.astype(np.float32)

        return points

    def get_normals(self, cloud):
        #return surface normals for all points in a cloud
        get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
        return get_normals_prox(cloud).cluster

    def return_color_list(self, cluster_count, color_list = []):
        #get a list of random colors to assign to segmentented clusters
        #list is only augmented if there are more colors than the length
        #of the existing list
        if (cluster_count > len(color_list)):
            for i in range(len(color_list), cluster_count):
                color_list.append(random_color_gen())
        return color_list

    def return_colorized_clusters(self, cloud, cluster_indices, color_list = []):
        #take in point cloud and array of point indices within separate clusters
        #return array of randomly colored point cloud arrays 
        colorized_clusters_list = []
        cluster_colors = self.return_color_list(len(cluster_indices), color_list)

        for j, indices in enumerate(cluster_indices):
            color = rgb_to_float(cluster_colors[j])
            for i in indices:
                colorized_clusters_list.append([cloud[i][0], cloud[i][1], cloud[i][2], color])

        colorized_clusters = pcl.PointCloud_PointXYZRGB()
        colorized_clusters.from_list(colorized_clusters_list)

        return colorized_clusters




