[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)
#  Project: Perception Pick & Place
---
[//]: # (Image References)
[image1]: ./pics/pl1.png
[image2]: ./pics/pl2.png
[image3]: ./pics/pl3.png


### Perception steps
In order to work with the point-cloud-data we first need to convert it from ros to pcl
```python 
cloud = ros_to_pcl(pcl_msg)
```
The first step after that was to elimante the noise, by using a statistical outlier filter we cann achieve the desired effect.
```python 
outlier_filter = cloud.make_statistical_outlier_filter()
outlier_filter.set_mean_k(50)
outlier_filter.set_std_dev_mul_thresh(1.0)
cloud = outlier_filter.filter()
```
After that a downsampling of the point cloud data was required to work in a resource efficient way with the pcl-data
```python 
    LEAF_SIZE = 0.01
    vox = cloud.make_voxel_grid_filter()
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud = vox.filter()
```
By adjusting the `LEAF_SIZE` we can adjust the result of the voxel grid filter

Now we just need to add a passthrough filter and a RANSAC Segmentation

```python
# PassThrough Filter
FILTER_AXIS = 'z'
AXIS_MIN = 0.6
AXIS_MAX = 1.1
passthrough = cloud.make_passthrough_filter()
passthrough.set_filter_field_name(FILTER_AXIS)
passthrough.set_filter_limits(AXIS_MIN, AXIS_MAX)
cloud = passthrough.filter()

# RANSAC Plane Segmentation
MAX_DISTANCE = 0.01
seg = cloud.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(MAX_DISTANCE)
inliers, _ = seg.segment()

# Extract inliers and outliers
cloud_objects = cloud.extract(inliers, negative=True)
cloud_table = cloud.extract(inliers, negative=False)
```
With the extracted `cloud_objects` we can now move on to clustering

### Clustering
To use the euclidean clustering method we first need to generate a `kd-Tree` to do so we need a "white-cloud", meaning a point-cloud with just spatial information no RGB-Data.
```python
white_cloud = XYZRGB_to_XYZ(cloud_objects)
tree = white_cloud.make_kdtree()
```
Now we can extract our clustered data like this
```python
ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.012)
ec.set_MinClusterSize(100)
ec.set_MaxClusterSize(15000)
ec.set_SearchMethod(tree)
cluster_idx = ec.Extract()
```
and finally colorize it for visual purposes with random, different colors and publish the pcl-data.
```python
# Create Cluster-Mask Point Cloud to visualize each cluster separately
cluster_color = get_color_list(len(cluster_idx))
color_cluster_point_list = []

for j, idx in enumerate(cluster_idx):
    for i, ix in enumerate(idx):
        color_cluster_point_list.append([white_cloud[ix][0], white_cloud[ix][1], white_cloud[ix][2],rgb_to_float(cluster_color[j])]) 

cloud = pcl.PointCloud_PointXYZRGB()
cloud.from_list(color_cluster_point_list)

# Convert PCL data to ROS messages
cloud_pcl = pcl_to_ros(cloud)

# Publish ROS messages
pcl_cloud_pub.publish(cloud_pcl)
```
With the now clustered data we cann continue with to the next step.

### Object recognition
With the previus trained SVM with the help of the `capture_feature.py` and the `train_SVM.py` scripts we can now load our trained model `model.sav` in our current script like this
```python
model = pickle.load(open('model.sav', 'rb'))
clf = model['classifier']
encoder = LabelEncoder()
encoder.classes_ = model['classes']
scaler = model['scaler']
```
When loaded we can use the `clf` object while iterating over the list of clustered objects.
But first we need to get the histograms / features of each object.
```python
chists = compute_color_histograms(ros_cluster, using_hsv=True)
normals = get_normals(ros_cluster)
nhists = compute_normal_histograms(normals)
feature = np.concatenate((chists, nhists))
```
Now we can run a prediction like this.
```python
prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
label = encoder.inverse_transform(prediction)[0]
detected_objects_labels.append(label)
```
Finally we publish the label of the prediction and append our `DetectedObject` to a list of those.
```python
# Publish a label
label_pos = list(white_cloud[pts_list[0]])
label_pos[2] += .4
object_markers_pub.publish(make_label(label,label_pos, index))

# Add the detected object to the list of detected objects.
do = DetectedObject()
do.label = label
do.cloud = ros_cluster
detected_objects.append(do)
```
This list of objects gets passed to the `pr2_mover` function where we handle the last steps of this project.

### Pick and Place Setup
As a first step after the initialization of the ros-variables we extract the pick-list from the ros-server.
```python
object_list_param = rospy.get_param('/object_list')
```
Now we loop through the entire pick list and specifiy the corresponding data for each object in the list.
```python
for i in range(0, len(object_list_param)):
	object_name.data = object_list_param[i]['name']
	arm_name.data = object_list_param[i]['group']    
```
To specify which arm and box whould be used we simply go over this if-statement with each iteration
```python
if arm_name.data == 'green':
        arm_name.data = 'right'
        place_pose.position.x = 0.0
        place_pose.position.y = -0.71
        place_pose.position.z = 0.605
    else:
        arm_name.data = 'left'  
        place_pose.position.x = 0.0
        place_pose.position.y = 0.71
        place_pose.position.z = 0.605
```
At this point all that is left to complete the required data for the `pick_place_routine` is to get the centroid of the pick-object. To do so we just need to find the cluster in out pcl-data that is corresponding to our current object of the picklist.
```python
for obj in objects:
    if obj.label == object_name.data:
        print ('Found picklist object in PCL data')
        # Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(obj.cloud).to_array()
        cent = np.mean(points_arr, axis=0)[:3]
        # Create 'place_pose' for the object
        pick_pose.position.x = np.asscalar(cent[0])
        pick_pose.position.y = np.asscalar(cent[1])
        pick_pose.position.z = np.asscalar(cent[2])
```
Now we have all the data we need and can add the dict to our `.yaml` file and run the `pick_place_routine`.
```python
yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
dict_list.append(yaml_dict)
.
.
.
resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
```
The last step is to save the `output_n.yaml` file.
```python
file = 'output_' + str(test_scene_num.data)
    send_to_yaml(file, dict_list)
```

### Results
#### World1 / Picklist1 100% (3/3)
![alt text][image1]

#### World2 / Picklist2 100% (5/5)
![alt text][image2]

#### World2 / Picklist2 87.5% (7/8)
![alt text][image3]




