[//]: # (Image References)
[normalized_confusion_matrix]: ./images/Normalized_confusion_matrix.png
[raw_confusion_matrix]: ./images/Raw_confusion_matrix.png
[shadow_puppets]: ./images/shadow_puppets.png
[test_world_1_result]: ./images/test_world_1.png
[test_world_2_result]: ./images/test_world_2.png
[test_world_3_result]: ./images/test_world_3.png

# Project: Perception Pick & Place

## 1. Structure of main program
 My first step in this project was to improve the organization and modularity of the project code. I chose to break the primary functions of the code into a few classes:
 * `PR2()`: This primary class contains most of the functions of the robot, including the `pcl_callback()` method, which responds to new pointcloud messages from the `/pr2/world/points` topic.
 * `Segmenter()`: This class contains all of the perception methods required to process the point cloud, detect objects, and publish object markers. It is instantiated and utilized within `PR2()`. `Segmenter()` is kept in a separate file from the main python file from the main PR2 file.
 * `dropbox_data()`: A container class for the pick and place dropbox requests passed to the robot from the `/dropbox` topic
 * `pick_place_data()`: A container class for the pick and place requests that are passed from the robot to `pick_place_routine` service in order to pick and place detected objects
 
 When the PR2.py script starts up, it instantiates the `PR2()` class and passes incoming pointcloud messages to `PR2.pcl_callback()`.
 
 ```python
 def main():
    model_file = '../training/model_100_orientations_sigmoid_YCbCr_16_bin.sav'

    # ROS node initialization
    rospy.init_node('pr2', anonymous=True)

    pr2 = PR2(model_file)

    #initialize point cloud subscriber
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pr2.pcl_callback, queue_size=1)

    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except:
        pass
```

## 2. Perception pipeline and segmenter class
 ### a. Filtering, RANSAC plane filtering, object clustering
  In order to interpret the point cloud message, significant processing was required. I tackled this problem by making separate methods for all of the processing steps that are required in the pick and place project, then calling them in the appropriate order and with experimentally determined parameters in order to achieve the desired result: accurate segmentation of object clusters.
  
  The order of operations implemented in `PR2.segment_scene()` is as follows:
   * Initialize the `Segmenter()` class
   * Convert the ROS point cloud message to point cloud library format
   * Downsample the point cloud to a more manageable density using `Segmenter.voxel_grid_downsample(cloud, leaf_size)`
   * Remove outliers from the point cloud using `Segmenter.outlier_filter(cloud, n_neighbors, threshhold_scale)`
   * Apply a passthrough filter to remove points below the table and the front edge of the table using `Segmenter.axis_passthrough_filter(self, cloud, axis, bounds)`
   * Use RANSAC plane segmentation to separate the table surface points from object points using `Segmenter.ransac_plane_segmentation(cloud, max_distance)`
   * Apply a secondary outlier filter to the object clouds
   * Separate the objects cloud into separate object clouds with Euclidean clustering using `Segmenter.get_euclidean_cluster_indices(self, cloud, tolerance, size_bounds)`
   * Detect objects with the pre-trained SVM model using `Segmenter.detect_objects(cloud, cluster_indices)`
   * Publish the results of the object detection to RViz labels using `Segmenter.publish_detected_objects(detected_objects, marker_pub, objects_pub)`
   * Set class variables with the detection results for later use
   
  Note: some lines are removed/modified in `segment_scene()` for clarity.
  ```python
  def segment_scene(self, pcl_msg):
        seg = self.segmenter #to reduce verbosity below

        # Convert ROS msg to PCL data
        cloud = ros_to_pcl(pcl_msg)
        leaf_size = 0.005

        # Voxel Grid Downsampling
        cloud = seg.voxel_grid_downsample(cloud, leaf_size = leaf_size)

        # Reduce outlier noise in object cloud
        cloud = seg.outlier_filter(cloud, 15, 0.01)
        # Save this to publish to RViz
        denoised_cloud = cloud

        # Passthrough Filter
        cloud = seg.axis_passthrough_filter(cloud, 'z', (0.55, 2)) #filter below table
        cloud = seg.axis_passthrough_filter(cloud, 'x', (.35, 10)) #filter out table front edge

        # RANSAC Plane Segmentation
        # Extract inliers and outliers
        table_cloud, objects_cloud = seg.ransac_plane_segmentation(cloud, max_distance = leaf_size)

        #Reduce outlier noise in object cloud
        objects_cloud = seg.outlier_filter(objects_cloud, 10, 0.01)

        # Euclidean Clustering and Object Detection
        cluster_indices = seg.get_euclidean_cluster_indices(objects_cloud, 0.03, (10,5000))
        detected_objects, detected_objects_dict = seg.detect_objects(objects_cloud, cluster_indices)
        
        # Convert PCL data to ROS messages
        # Publish ROS messages
        message_pairs = [(denoised_cloud, self.denoised_pub),
                         (objects_cloud, self.objects_pub)
                         ]
        
        seg.convert_and_publish(message_pairs)

        #publish detected objects and labels
        seg.publish_detected_objects(detected_objects,
                                     self.object_markers_pub,
                                     self.detected_objects_pub)

        self.object_list = detected_objects_dict
        self.detected_objects = detected_objects
        self.table_cloud = table_cloud
   ```

 ### b. Feature Extraction and Object Detection SVM (including addition of additional color space, allowing for reducing bin count)
  Put make the sensor stick package a separate repo pulled into the pick and place project repo as a submodule rather than copying it into the perception project or having duplicated training/capture functions for the perception exercises and project.
  
  Within sensor_stick, I updated the feature extraction script (sensor_stick/src/sensor_stick/features.py) used to train my object detection SVM in a few key ways:
   1. Added YCbCr color histogram extraction to my feature vector
   2. Switched from RGB to HSV color histogram extraction
   3. Updated the number of histogram bins (normals, HSV color, and YCbCr color) to 16 instead of 32
   
  I saw dramatic improvement in the efficacy of my object detection model by using both HSV and YCbCr color spaces even using half as many histogram bins. Both of these color spaces allow for the separation of brightness from the color of the object, which helps remove ambiguity due to the position/orientation of an object relative to a light source (unlike RGB). In order to compensate for the addition of an additional histogram calculation, I reduced the number of bins in each histogram by half. To my surprise, this did not have a significant negative effect on the accuracy of my object detection model, though it did significantly increase the speed with which the histograms could be calculated.
  
  The relevant functions are:
  ```python
  def compute_color_histograms(cloud):
       # Compute histograms for the clusters
       point_colors_list = []
       points = pc2.read_points(cloud, skip_nans=True)

       # Step through each point in the point cloud
       for point in points:
           rgb_list = float_to_rgb(point[3])
           point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
           point_colors_list.append(rgb_to_YCbCr(rgb_list))

       normed_features = return_normalized_hist(point_colors_list, bins_range = (0,256))

       return normed_features
  
  def rgb_to_hsv(rgb_list):
       rgb_normalized = [1.0*rgb_list[0]/255, 1.0*rgb_list[1]/255, 1.0*rgb_list[2]/255]
       hsv_normalized = matplotlib.colors.rgb_to_hsv([[rgb_normalized]])[0][0]
       return hsv_normalized

  def rgb_to_YCbCr(rgb_list):
       r, g, b = rgb_list

       Y = 16 + (65.738*r + 129.057*g + 25.064*b)/256.
       Cb = 128 +  (-37.945*r - 74.203*g + 112.0*b)/256.
       Cr = 128 + (112.0*r - 93.786*g + 18.214*b)/256.

       YCbCr = [Y, Cb, Cr]

       return YCbCr
   
   def return_normalized_hist(features, bins_range, nbins = 16):
       hist = []
       features = np.asarray(features)
       length, depth = features.shape

       for i in range(depth):
           hist.extend(np.histogram(features[:,i], bins = nbins, range = bins_range)[0])

       hist = hist/np.sum(hist).astype(np.float)

       return hist
   ```
   
   I ultimate trained my object detection SVM using 100 random orientations of each object. After some experimentation, I chose to use a sigmoid kernel because it seemed to generate the most repeatable and accurate object labels. My confusion matrix results are shown below:
   
   ![Normalized Confusion Matrix][normalized_confusion_matrix]

## 3. Successful Object Detection in Each Test World
 My object detection SVM was relatively successful in identifying the objects in the 3 test worlds. I achieved the following correctly identified object detection rates:
  * Test World 1 ([see yaml](./output_1.yaml)): 3 of 3 (100%)
  * Test World 2 ([see yaml](./output_2.yaml)): 5 of 5 (100%)
  * Test World 3 ([see yaml](./output_3.yaml)): 7 of 8 (88%)
 
 Here are the camera views as captured in RViz:
  ![Test World 1][test_world_1_result]
  
  ![Test World 2][test_world_2_result]
    
  ![Test World 3][test_world_3_result]

## 4. Successful pick and place!
 With the collision map in place and a fairly accurate object detection SVM, I was able to successfully pick and place objects! Here's a video of a successful run.
 
 [![Successful Pick and Place Run](https://img.youtube.com/vi/bGgx0UMarA0/0.jpg)](https://www.youtube.com/watch?v=bGgx0UMarA0)
 
 Unfortunately, since I had trouble updating the collision map after each successive object was picked, I was only able to pick and place a single object with the collision map enabled. With the collision map disabled, I was able to pick multiple objects in a row, though there was some uncertaintly in whether they would make it to the drop box intact...
 
 I found that even when the correct grasp location was passed to the pick and place service, repeatability was a serious problem. I also was not able to test out the pick and place function on larger object sets due to performance limitations on my laptop. Gazebo + RViz is quite heavy, especially within the virtual machine (running on only 2 cores).

## 5. Building collision map
 I first chose to build the collision map without rotating the PR2 world joint. I chose to implement the collision map in two stages. First, I built a baseline map for the static objects that would not be grasped. In the simple case in which the robot does not rotate, this includes only the table. Then, each time I detected the next object in the pick list queue, I appended all other objects that had not yet been picked to the collision map cloud. I then published the collision cloud to the `/pr2/3d_map/points` topic. Unfortunately, for reasons I wasn't able to determine, the collision cloud did not update each time I published a new set of points.
 
 ```python
 def build_collision_map(self):
    # update this to include drop box object cloud as well
    # would need to train SVM for this
    self.collision_map_base_list.extend(list(self.table_cloud))

 def publish_collision_map(self,picked_objects):
     obstacle_cloud_list = self.collision_map_base_list

     for obj in self.detected_objects:
         if obj.label not in picked_objects:
             print('Adding {0} to collision map.'.format(obj.label))
             obj_cloud = ros_to_pcl(obj.cloud)
             obstacle_cloud_list.extend(list(obj_cloud))
             #print(obstacle_cloud)
         else:
             print('Not adding {0} to collision map because it appears in the picked object list'.format(obj.label))

      obstacle_cloud = pcl.PointCloud_PointXYZRGB()
      obstacle_cloud.from_list(obstacle_cloud_list)

      self.segmenter.convert_and_publish([(obstacle_cloud, self.collision_cloud_pub)]) 
   ```
   
   I was also able to implement a method to rotate the robot to look to the left and right at the side tables and drop boxes. I had planned on adding these to the base collision map, but I ran into a couple of issues. First, I hadn't originally trained the SVM to recognize the drop boxes, so I didn't have a quick way to recognize them and add them to the collison map. In addition, I realized that my assumptions in the segment scene method (e.g. distance to the front of the table for plane segmentation) may not be accurate for the side tables. In any case, once I saw that the paths generated by the pick and place service never really came close to the drop box, I decided there wasn't really a need to add the side table and drop box to the collision map.
   
   Here's my code for moving the robot, though... Each time the main script starts up, the robot will move itself according to `PR2.goal_positions`, an array of world joint orientations at which to observe the obstacles in the environment. Once all the goal positions have been achieved and the collision map observed for that orientation, the primary arm mover method is called to pick and place the objects.
   
   ```python
   if len(self.goal_positions)>0:
       new_position = self.goal_positions.pop()
       self.move_world_joint(new_position)

       #segment scene and detect objects
       self.segment_scene(pcl_msg)

       #add obstacles (except for recognized pick objects)
       #to the base collision map
       self.build_collision_map()
   else:
       #identify the objects listed in the pick list
       #submit them to the pick_place_routine
       self.mover()
   ```
   
   Here are the methods that actually move the robot joint.
   
   ```python
   def move_world_joint(self, goal):
       print('Attempting to move world joint to {}'.format(goal))
       pub_j1 = rospy.Publisher('/pr2/world_joint_controller/command',
                                Float64, queue_size=10)
       increments = 10
       position = self.get_world_joint_state()

       goal_positions = [n*1.0/increments*(goal-position) + position for n in range(1, increments + 1)]
       for i, g in enumerate(goal_positions):
           print('Publishing goal {0}: {1}'.format(i, g))
           while abs(position - g) > .005:
               pub_j1.publish(g)
               position = self.get_world_joint_state()
               print('Position: {0}, Error: {1}'.format(position, abs(position - g)))
               rospy.sleep(1)
    
     def get_world_joint_state(self):
       try:
           msg = rospy.wait_for_message("joint_states", JointState, timeout = 10)
           index = msg.name.index('world_joint')
           position = msg.position[index]
       except rospy.ServiceException as e:
           print('Failed to get world_joint position.')
           position = 10**6

       return position
   ``` 
## 6. Bloopers
 I noticed quite a few issues with the implementation of the pick and place robot project. For one, the simulation is really heavy, and I had a lot of trouble getting it to run successfully in the VM on my laptop (Macbook Air). I got around the issue by working in test world 1 for the most part since having fewer objects seemed to reduce the computation load significatly. However, when trying to rotate the robot, I routinely got down to <2 fps in the simulation.
 
 I also had problems getting the environment to launch successfully. A number of times I had to completely restart my VM after numerous unsuccessful attempts just to launch the program.
 
 Here's an example of one of the repeated issues along those lines. I'm not sure what gave it the idea, but it seems like PR2 was trying to make shadow puppets:
 
 ![Put a bird on it!][shadow_puppets]
 
 Dramatic reenactment:
 ![Colbert??][https://gph.is/29Xkwyz]
 
## 7. Improvements for future:
 There are still a number of things I would like to improve if I were to continue working on this project.
 
 1. Get a better computer:
  Not really a technical improvement, but I think I would have had a much easier time with a more powerful computer...
 2. Improve the collision map:
  I'd like to retrain my SVM to recognize the drop boxes so I could add them to the static collision map and get a fuller represenation of the map by rotating the robot around. In addition, I never figured out why the collision map wasn't updating in RViz even though I was sending it a new map with the next object to pick removed from the map.
 3. Change the PCL callback structure:
  There are some inherent limitations in the way the pcl callback method is written. It gets a single point cloud, which it uses to locate the objects in the scene and generate a collision map. Unfortunately, since the pick and place service isn't completely reliable, some objects may be moved, dropped, etc., resulting in inaccurate collision maps and grasp locations. I'd like to consider other workflows that would allow me to check whether the object was successfully dropped in the box, update the collision map and grasp locations if the objects move, and reattempt failed pick and place operations.
 4. Improve feature extraction efficiency:
  I realized that a significant portion of the computational load comes from generating the feature histograms for all of the object clusters in the scene. Buiding the histogram seems to be fairly inefficient process in Numpy, but it is made much more so by not vectorizing the operation. I didn't dig into it too much, but Numpy's `numpy.histogramdd()` seems like it would be much more efficient than an iterative approach. 



