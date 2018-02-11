#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml

def mover(j):
    pub_j1 = rospy.Publisher('/pr2/world_joint_controller/command', Float64, queue_size=1)
    pub_j1.publish(j*(np.pi/180))


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# PERCEPTION

    # Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)
    
    # Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50)
    outlier_filter.set_std_dev_mul_thresh(1.0)
    cloud = outlier_filter.filter()

    # Voxel Grid Downsampling
    LEAF_SIZE = 0.01
    vox = cloud.make_voxel_grid_filter()
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud = vox.filter()

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

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # set param
    ec.set_ClusterTolerance(0.012)
    ec.set_MinClusterSize(100)
    ec.set_MaxClusterSize(15000)
    # search for clusters
    ec.set_SearchMethod(tree)
    cluster_idx = ec.Extract()

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

# OBJECT RECOGNITION

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_idx):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)

        # Convert the cluster from pcl to ROS 
        ros_cluster = pcl_to_ros(pcl_cluster)   

        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)

        # Concatenate the feature vector
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(objects):

    # Initialize variables
    dict_list = []
    test_scene_num = Int32()
    test_scene_num.data = 3
    object_name = String()
    arm_name = String()
    pick_pose = Pose()
    place_pose = Pose()

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    print('Picklist:')
    for i in range(0, len(object_list_param)):
        print '-item_'+str(i)+':', object_list_param[i]['name']

    # Rotate PR2 in place to capture side tables for the collision map
    mover(0)

    # Loop through the pick list
    for i in range(0, len(object_list_param)):
        object_name.data = object_list_param[i]['name']
        arm_name.data = object_list_param[i]['group']        
        # Assign the arm to be used for pick_place
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
            

        # search objects for label
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
                print object_name.data, ' matches ', obj.label
                print('Pick Position (x,y,z)')
                print(pick_pose.position.x ,pick_pose.position.y ,pick_pose.position.z)
                print('Place Position (x,y,z)')
                print(place_pose.position.x ,place_pose.position.y ,place_pose.position.z)
              

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        dict_list.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            #pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            #resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            #print ("Response: ",resp.success)
            pass
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    # Output your request parameters into output yaml file
    file = 'output_' + str(test_scene_num.data)
    send_to_yaml(file, dict_list)


if __name__ == '__main__':
     # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)

    # Create Subscribers
    pcl_sub = rospy.Subscriber('/pr2/world/points', pc2.PointCloud2, pcl_callback, queue_size=1)

    # publisher objects to publish pcl-data for the table and the objects ontop
    pcl_cloud_pub = rospy.Publisher('/cloud', PointCloud2, queue_size=1)
    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size=1)
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size=1)

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
