from styx_msgs.msg import TrafficLight
from traffic_l.model import detect_traffic_lights,read_traffic_lights
import rospy
import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
# from matplotlib import pyplot as plt
from PIL import Image
from os import path
from traffic_l.utils import label_map_util
# from utils import visualization_utils as vis_util
import time
import cv2
import math
class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        Num_images = 17
        PATH_TO_TEST_IMAGES_DIR = './test_images'
        MODEL_NAME = 'traffic_l/faster_rcnn_resnet101_coco_11_06_2017'
        self.sess,self.image_tensor,self.detection_boxes,self.detection_scores,self.detection_classes,self.num_detections=detect_traffic_lights(PATH_TO_TEST_IMAGES_DIR, MODEL_NAME, Num_images, plot_flag=False,transfer_seesion=True)
        
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # image = Image.open(image_path)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # image_np = load_image_into_numpy_array(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image_np = np.asarray(image, dtype=np.uint8)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})


        red_flag = read_traffic_lights(image_np, np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32))
        
        rospy.loginfo("RedFlag"+str(red_flag))
        if red_flag:
            return TrafficLight.RED
        else:
            return TrafficLight.GREEN
        return 