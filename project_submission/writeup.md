## Project: Perception Pick & Place
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


# Required Steps for a Passing Submission:
1. Extract features and train an SVM model on new objects (see `pick_list_*.yaml` in `/pr2_robot/config/` for the list of models you'll be trying to identify). 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  [See the example `output.yaml` for details on what the output should look like.](https://github.com/udacity/RoboND-Perception-Project/blob/master/pr2_robot/config/output.yaml)  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.
9. Congratulations!  Your Done!

# Extra Challenges: Complete the Pick & Place
7. To create a collision map, publish a point cloud to the `/pr2/3d_map/points` topic and make sure you change the `point_cloud_topic` to `/pr2/3d_map/points` in `sensors.yaml` in the `/pr2_robot/config/` directory. This topic is read by Moveit!, which uses this point cloud input to generate a collision map, allowing the robot to plan its trajectory.  Keep in mind that later when you go to pick up an object, you must first remove it from this point cloud so it is removed from the collision map!
8. Rotate the robot to generate collision map of table sides. This can be accomplished by publishing joint angle value(in radians) to `/pr2/world_joint_controller/command`
9. Rotate the robot back to its original state.
10. Create a ROS Client for the “pick_place_routine” rosservice.  In the required steps above, you already created the messages you need to use this service. Checkout the [PickPlace.srv](https://github.com/udacity/RoboND-Perception-Project/tree/master/pr2_robot/srv) file to find out what arguments you must pass to this service.
11. If everything was done correctly, when you pass the appropriate messages to the `pick_place_routine` service, the selected arm will perform pick and place operation and display trajectory in the RViz window
12. Place all the objects from your pick list in their respective dropoff box and you have completed the challenge!
13. Looking for a bigger challenge?  Load up the `challenge.world` scenario and see if you can get your perception pipeline working there!

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
[//]: # (Image References)
[normalized_confusion_matrix]: ./images/Normalized_confusion_matrix.png
[raw_confusion_matrix]: ./images/Raw_confusion_matrix.png
[shadow_puppets]: ./images/shadow_puppets.png
[test_world_1_result]: ./images/test_world_1.png
[test_world_2_result]: ./images/test_world_2.png
[test_world_3_result]: ./images/test_world_3.png

[![Successful Pick and Place Run](https://img.youtube.com/vi/bGgx0UMarA0/0.jpg)](https://www.youtube.com/watch?v=bGgx0UMarA0)

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
  * Test World 1: 3 of 3 (100%) 
  * Test World 2: 5 of 5 (100%)
  * Test World 3: 7 of 8 (88%)
  
 Here are the camera views as captured in RViz:
  ![Test World 1][test_world_1_result]
  
  ![Test World 2][test_world_2_result]
    
  ![Test World 3][test_world_3_result]
  
 
 ### a. images of each result with links to the yaml outputs
## 4. Successful pick and place!
## 5. Building collision map
 ### a. Would have added boxes to map, but needed to train SVM on boxes
 ### b. Implemented turn and look, but too slow to run routinely, and didn't have problems colliding with boxes anyway
 ### c. Had trouble getting the collision map to update each iteration even though I was sending a new map to the topic
## 6. Bugginess of program overall, including shadow puppet pose
## 7. Improvements for future:
 ### a. use a more powerful computer to make things less painful
 ### b. retrain SVM to recognize the red and blue boxes in addition to the other objects
 ### c. Implement the turn and look function to build up the baseline collision map including the red and blue boxes
 ### d. Investigate why the collision map does not update correctly after each object is picked
 ### e. Learn to determine whether the object was successfully dropped in the box, and attempt to pick it up again if not
  #### i. would require a change in architecture since now pointcloud is only passed in before the pick and place routine starts
 ### f. Improve the efficiency of the capture script by vectorizing the histogram processing in numpy


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



