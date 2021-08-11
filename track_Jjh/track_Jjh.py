#!/usr/bin/env python
# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg


import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2, time
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
import numpy as np
import os.path
import time
import time
import rospy
import rospkg
import cv_bridge
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import String
from geometry_msgs.msg import Point , PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge, CvBridgeError
def find_delta(image_shape, going_pixels):

    ############################# angle1 ###################################
    #print(going_pixels[0]-(image_shape[1]/2))
    angle_radian = math.atan(((going_pixels[0]-(image_shape[1]/2))/2)/(image_shape[0]-going_pixels[1])) ## /2
    angle_degree = angle_radian * (180/np.pi)

    #print('mid x1', mid_x1, 'mid x2', mid_x2, 'y1', y1, 'y2', y2)
    #print('radian1: ',angle_radian, 'degree1', angle_degree)

    ############################# pid1 ###################################

    fontType = cv2.FONT_HERSHEY_SIMPLEX
    p = 0.17


    degree = "left" if angle_degree < 0 else "right"
    degree_text = str(round(abs(angle_degree), 3)) + '. ' + degree
    cv2.putText(line_image, degree_text, (30, 100), fontType, 1., (255, 255, 255), 3)

    ########################### distance ######################################
    print(going_pixels[0],image_shape[1]/2)
    direction = "left" if going_pixels[0]-(image_shape[1]/2) < 0 else "right"
    deviation_text = 'target is ' + str(round(abs(going_pixels[0]-(image_shape[1]/2)), 3)) + 'pixel ' + direction + ' of center'
    cv2.putText(line_image, deviation_text, (30, 150), fontType, 1., (255, 255, 255), 3)

    ########################## FPS ####################################################

    # deviation_text = str(round(fps, 3)) + ' FPS'
    # cv2.putText(line_image, deviation_text, (30, 50), fontType, 1., (255, 255, 255), 3)

    return line_image, -angle_degree*p

def cal_boxsize(xmin,ymin,xmax,ymax):
    x_length = xmax - xmin
    y_length = ymax - ymin
    boxsize = x_length * y_length
    return boxsize

def image_callback(data):
    global image_shape
    try: # Read the image from the input topic:
        cv_image = bridge.imgmsg_to_cv2(data)
    except CvBridgeError, e:
            print e
    frame = np.zeros((cv_image.shape[0],cv_image.shape[1],3),np.uint8)
    frame[:,:,0] = cv_image
    frame[:,:,1] = cv_image
    frame[:,:,2] = cv_image
    image_shape=[frame.shape[0],frame.shape[1]]
    

def BoundingBoxes_callback(data):
    global BoundingBox
    cone_xmin = data.bounding_boxes[0].xmin
    cone_ymin = data.bounding_boxes[0].ymin #ymin
    cone_xmax = data.bounding_boxes[0].xmax #xmax
    cone_ymax = data.bounding_boxes[0].ymax #ymax
    cone_class = data.bounding_boxes[0].Class
    cone_y_center = (cone_ymax+cone_ymin)/2
    cone_x_center = (cone_xmax+cone_xmin)/2
    cone_box_size = cal_boxsize(cone_xmin,cone_ymin,cone_xmax,cone_ymax)
    BoundingBox =  [cone_xmin,cone_ymin,cone_xmax,cone_ymax,cone_y_center,cone_x_center,cone_box_size,cone_class]

def Cone_information(data):
    Blue_informations = []
    Yello_informations = []
    Blue_information = []
    Yello_information = []
    Class=data[7]
    if (Class=='blue'):
        blue_cone_xmin,blue_cone_ymin,blue_cone_xmax,blue_cone_ymax,blue_cone_y_center,blue_cone_x_center,blue_cone_box_size = data[0:7]
        Blue_information = [blue_cone_xmin,blue_cone_ymin,blue_cone_xmax,blue_cone_ymax,blue_cone_y_center,blue_cone_x_center,blue_cone_box_size]
        Blue_informations.append(Blue_information)
    elif (Class=='yellow'):
       [yello_cone_xmin,yello_cone_ymin,yello_cone_xmax,yello_cone_ymax,yello_cone_y_center,yello_cone_x_center,yello_cone_box_size] = data[0:7]
       Yello_information = [yello_cone_xmin,yello_cone_ymin,yello_cone_xmax,yello_cone_ymax,yello_cone_y_center,yello_cone_x_center,yello_cone_box_size]
       Yello_informations.append(Yello_information)
    print(Blue_informations,Yello_informations)
    return [Blue_informations,Yello_informations]

def Select_biggest_box(data):
    box_size_max = np.array(data)[:,6].max() #6 is boxsize collunm
    box_size_max_num = np.wehre(np.array(data)[:,6]==box_size_max)
    biggest_box_information = np.array(data)[box_size_max_num,:]
    return biggest_box_information
    
def Make_pixel(box1_data,box2_data): #make target coordinates
    x_cordination = (box1_data[5] + box2_data[5])/2
    y_cordination = (box1_data[4] + box2_data[4])/2
    going_pixels = [x_cordination,y_cordination]
    return going_pixels

if __name__ == '__main__':
    
    rospy.init_node('Track_mission', anonymous=True)
    bridge = CvBridge()
    rospy.Subscriber("darknet_ros/detection_image", Image, image_callback)
    rospy.Subscriber("darknet_ros/bounding_boxes", BoundingBoxes, BoundingBoxes_callback)
    image_publisher = rospy.Publisher("/final_image", Image, queue_size=100)
    control_publisher = rospy.Publisher("/Lane_ack_vel", AckermannDriveStamped, queue_size=100)
    rate = rospy.Rate(10)
    ackermann_cmd = AckermannDriveStamped()
    while (True):
        try:
            [Blue_informations,Yello_informations] = Cone_information(BoundingBox)
            Blue_biggest_box = Select_biggest_box(Blue_informations)
            Yello_biggest_box = Select_biggest_box(Yello_informations) #make_point
            going_pixels = Make_pixel(Blue_biggest_box,Yello_biggest_box)
            final_image, image_delta = find_delta(image_shape,going_pixels) #make_delta
            print(image_delta)

            if (image_delta>28):
                image_delta=28
            elif(image_delta<-28):
                image_delta=-28

            L=1.3
            ld=2.5
            final_image_theta=math.atan(2*L*math.sin(image_delta*np.pi/180)/ld)*(180/np.pi)
            ackermann_cmd.drive.steering_angle = final_image_theta * math.pi / 180
            ackermann_cmd.drive.speed = 3.5
            control_publisher.publish(ackermann_cmd) #publish
            image_publisher.publish(bridge.cv2_to_imgmsg(final_image)) #publish
            if cv2.waitKey(1) == ord('q'):
                break

        except rospy.ROSInterruptException:
            pass