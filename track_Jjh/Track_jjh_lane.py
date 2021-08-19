#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# @Author  : Luo Yao
# @Modified  : AdamShan
# @Original site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_node.py


import time
import math
import tensorflow as tf
import numpy as np
import cv2

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from config import global_config

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from lane_detector.msg import Lane_Image


CFG = global_config.cfg


class lanenet_detector():

    def __init__(self):
        # tf.enable_eager_execution()
        print("init")
        self.image_topic = rospy.get_param('~image_topic')
        self.output_image = rospy.get_param('~output_image')
        self.output_lane = rospy.get_param('~output_lane')
        self.lane_image_topic = rospy.get_param('~lane_image_topic')

        self.bridge = CvBridge()
        sub_image = rospy.Subscriber(self.image_topic, Image, self.img_callback, queue_size=1)
        self.pub_image = rospy.Publisher(self.output_image, Image, queue_size=1)
        self.pub_bin_image = rospy.Publisher('lane_bin', Image, queue_size=1)
        # self.pub_ist_image = rospy.Publisher(self.output_image, Image, queue_size=1)

    
    def img_callback(self, data):
        # print(time.time())
        t1 = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.namedWindow("ss")
        # cv2.imshow("ss", cv_image)
        # cv2.waitKey(0)
        original_img = cv_image.copy()
        resized_image = self.preprocessing(cv_image)
        mask_image = self.inference_net(resized_image, original_img)
        mask_image2 = cv2.resize(mask_image, (640, 480), interpolation=cv2.INTER_LINEAR)
        out_img_msg = self.bridge.cv2_to_imgmsg(mask_image2, "32FC1")
        self.pub_image.publish(out_img_msg)
        print(1/(time.time() - t1))
        #print(time.time())
        
    def preprocessing(self, img):
        image = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5
        # cv2.namedWindow("ss")
        # cv2.imshow("ss", image)
        # cv2.waitKey(1)
        return image

    def inference_net(self, img, original_img):

        binary_seg_image, instance_seg_image = self.sess.run([self.binary_seg_ret, self.instance_seg_ret],
                                                        feed_dict={self.input_tensor: [img]})               
        # cv2.namedWindow("ss")
        # cv2.imshow("ss", instance_seg_image[0])
        binary = binary_seg_image[0].astype(np.float32)

        return binary * 255


    def minmax_scale(self, input_arr):
        """

        :param input_arr:
        :return:
        """
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr



if __name__ == '__main__':
    # init args
    rospy.init_node('lanenet_node')
    lanenet_detector()
    rospy.spin()
