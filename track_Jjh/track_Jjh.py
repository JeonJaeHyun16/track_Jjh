#!/usr/bin/env python
# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv_bridge
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2, time
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
import numpy as np
import os.path
import time
import rospy
import rospkg
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import String
from geometry_msgs.msg import Point , PoseStamped


def cal_boxsize(xmin,ymin,xmax,ymax):
    x_length = xmax - xmin
    y_length = ymax - ymin
    boxsize = x_length * y_length
    return boxsize

def image_callback(data):
    cv_image = self.bridge.imgmsg_to_cv2(data)
    return cv_image

def BoundingBoxes_callback(data):
    cone_xmin = data.bounding_boxes[0].xmin
    cone_ymin=data.bounding_boxes[0].ymin #ymin
    cone_xmax=data.bounding_boxes[0].xmax #xmax
    cone_ymax=data.bounding_boxes[0].ymax #ymax
    cone_y_center = (ymax+ymin)/2
    cone_x_center = (xmax+xmin)/2
    cone_box_size = cal_boxsize(xmin,ymin,xmax,ymax)
    return  [cone_xmin,cone_ymin,cone_xmax,cone_ymax,cone_y_center,cone_x_center,cone_box_size]

def Cone_information(data):
    Blue_informations = []
    Yello_informations = []
    Blue_information = []
    Yello_information = []
    Class=data.bounding_boxes[0].Class
    if (Class=='Blue_cone'):
        blue_cone_xmin,blue_cone_ymin,blue_cone_xmax,blue_cone_ymax,blue_cone_y_center,blue_cone_x_center,blue_cone_box_size = Cone_information(data)
        Blue_information = [blue_cone_xmin,blue_cone_ymin,blue_cone_xmax,blue_cone_ymax,blue_cone_y_center,blue_cone_x_center,blue_cone_box_size]
        Blue_informations.append(Blue_information)
    elif (Class=='Yello_cone'):
       [yello_cone_xmin,yello_cone_ymin,yello_cone_xmax,yello_cone_ymax,yello_cone_y_center,yello_cone_x_center,yello_cone_box_size] = Cone_information(data)
       Yello_information = [yello_cone_xmin,yello_cone_ymin,yello_cone_xmax,yello_cone_ymax,yello_cone_y_center,yello_cone_x_center,yello_cone_box_size]
       Yello_informations.append(Yello_information)
    print(Blue_informations,Yello_informations)
    return [Blue_informations,Yello_informations]




if __name__ == '__main__':
    
    rospy.init_node('Track_mission', anonymous=True)
    Ori_Image=rospy.Subscriber("/darknet_ros/detection_image", Image, image_callback)
    BoundingBoxes =rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, BoundingBoxes_callback)
    pub_traffic = rospy.Publisher('/detect/traffic_sign', Int32MultiArray, queue_size=10)
    pub_delivery = rospy.Publisher('/delivery_zone', Int16MultiArray, queue_size=10)
    rate = rospy.Rate(10)
    
    traffic_array=Int32MultiArray()
    traffic_array.data=[0,0,0,0]
    delivery_alpha_A = String()
    delivery_alpha_A = 'A0'
    delivery_alpha_B = String()
    delivery_alpha_B = 'B0'
    delivery_array=Int16MultiArray()
    delivery_array.data=[1,1,1]

    while (True):
        try:
            [Blue_informations,Yello_informations] = Cone_information(BoundingBoxes)
            pub_traffic_sign(label)
            pub_delivery_sign_A(label)
            pub_delivery_sign_B(label)
            A_and_B_compare(label)
            pub_sign=False
            pub_sign2=False
            pub_sign3=False
            if cv2.waitKey(1) == ord('q'):
                break

        except rospy.ROSInterruptException:
            pass