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

def voxel_downsampling(cloud):
    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.008
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    return vox.filter()    

def filter_passthrough_zy(cloud):
    # TODO: PassThrough Filter
    # Passthrough z axis
    passthrough = cloud.make_passthrough_filter()
    filter_axis = 'z'
    axis_min = 0.6
    axis_max = 1.2
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    # Passthrough y axis
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'y'
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    cloud_filtered = passthrough.filter()
    return cloud_filtered

def filter_outlier_removal(cloud):
    # TODO: Outlier Removal Filter
    outlier_filter = cloud.make_statistical_outlier_filter()
    # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(40)
    # Set threshold scale factor
    x = 0.1
    # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
    # Finally call the filter function for magic
    cloud_filtered = outlier_filter.filter()
    return cloud_filtered

def RANSAC_extract_objects(cloud):
    # TODO: RANSAC Plane Segmentation
    seg = cloud.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = .01  # 0.04 to get rid of table edge
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
    
    # TODO: Extract inliers and outliers
    cloud_objects = cloud.extract(inliers, negative=True)
    return cloud_objects

def cluster_objects_Euclidean(cloud_objects):
    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = white_cloud.make_EuclideanClusterExtraction()

    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(4000)

    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()
    return cluster_indices, white_cloud

def mark_objects_with_color(cluster_indices, white_cloud):
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    return cluster_cloud

def label_object(ros_cluster):
    # Compute the associated feature vector
    chists = compute_color_histograms(ros_cluster, using_hsv=True)
    normals = get_normals(ros_cluster)
    nhists = compute_normal_histograms(normals)
    feature = np.concatenate((chists, nhists))
    # Make the prediction
    prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
    label = encoder.inverse_transform(prediction)[0]
    return label

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # TODO: Voxel Grid Downsampling
    cloud_filtered = voxel_downsampling(cloud)

    # TODO: PassThrough Filter
    cloud_filtered = filter_passthrough_zy(cloud_filtered)

    # # TODO: Outlier Removal Filter
    cloud_filtered = filter_outlier_removal(cloud_filtered)

    # TODO: RANSAC Plane Segmentation
    cloud_objects = RANSAC_extract_objects(cloud_filtered)

    # # TODO: Euclidean Clustering
    cluster_indices, white_cloud = cluster_objects_Euclidean(cloud_objects)
    # cluster_cloud = mark_objects_with_color(cluster_indices, white_cloud)

    # TODO: Convert PCL data to ROS messages
    # ros_cloud_objects = pcl_to_ros(cluster_cloud)

    # TODO: Publish ROS messages
    # pcl_objects_pub.publish(ros_cloud_objects)

# Exercise-3 TODOs:
    detected_objects_labels = []
    detected_objects = []
    # # Classify the clusters! (loop through each detected cluster one at a time)

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Compute the associated feature vector
        # Make the prediction
        label = label_object(ros_cluster)
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        # pr2_mover(detected_objects_list)
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    dict_list = []
    labels = []
    centroids = [] # to be list of tuples (x, y, z)
    
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox_list = rospy.get_param('/dropbox')

    # TODO: Parse parameters into individual variables
    object_names = [object_param['name'] for object_param in object_list_param]
    object_groups = [object_param['group'] for object_param in object_list_param]
    dropbox_groups = [box['group'] for box in dropbox_list]
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    for detected_object in object_list:
        labels.append(detected_object.label)
        object_idx = 0
        try:
            object_idx = object_names.index(detected_object.label)
        except rospy.ROSInterruptException:
            pass
        
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        points_arr = ros_to_pcl(detected_object.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3]
        centroid = [np.asscalar(x) for x in centroid]
        centroids.append(centroid)
        # TODO: Create 'place_pose' for the object
        PICK_POSE = Pose()
        PICK_POSE.position.x = centroid[0]
        PICK_POSE.position.y = centroid[1]
        PICK_POSE.position.z = centroid[2]

        # TODO: Assign the arm to be used for pick_place
        OBJECT_NAME = String()
        OBJECT_NAME.data = object_names[object_idx]

        WHICH_ARM = String()
        PLACE_POSE = Pose()
        box_id = dropbox_groups.index(object_groups[object_idx])
        WHICH_ARM.data = dropbox_list[box_id]['name']
        PLACE_POSE.position.x = dropbox_list[box_id]['position'][0]
        PLACE_POSE.position.y = dropbox_list[box_id]['position'][1]
        PLACE_POSE.position.z = dropbox_list[box_id]['position'][2]

        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(TEST_SCENE_NUM, WHICH_ARM, OBJECT_NAME, PICK_POSE, PLACE_POSE)
        dict_list.append(yaml_dict)
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        # try:
        #     pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

        #     # TODO: Insert your message variables to be sent as a service request
        #     resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

        #     print ("Response: ",resp.success)

        # except rospy.ServiceException, e:
        #     print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml(yaml_output_filename, dict_list)


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('recognition', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber('/pr2/world/points', pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher('/pcl_objects', PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher('/object_markers', Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher('/detected_objects', DetectedObjectsArray, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    TEST_SCENE_NUM = Int32()
    TEST_SCENE_NUM.data = 3
    yaml_output_filename = 'output_3.yaml'
    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()