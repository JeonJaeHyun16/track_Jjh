#!/usr/bin/env python

import cv2
import numpy as np
import roslib
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose
from ackermann_msgs.msg import AckermannDriveStamped
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import time
import sys
import os.path
import rospkg
from darknet_ros_msgs.msg import BoundingBoxes
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import MultiArrayLayout
from geometry_msgs.msg import Point , PoseStamped


def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))

def deg_to_rad(deg):
    return deg * math.pi / 180.0

def rad_to_deg(rad):
    return rad * 180.0 / math.pi


def WarpPerspecitve(img):

    img_size = (img.shape[1], img.shape[0]/3)

    height = img.shape[0]
    width = img.shape[1]

    # src = np.float32([(0, height/3),
    # (width, height/3),
    # (width, height),
    # (0, height)])


    # dst = np.float32([(width/4, height/4),
    # (width*3/4, height/4),
    # (width*3/4, height*3/4),
    # (width/4, height*3/4)])


    src = np.float32([(0, height/3),
    (width, height/3),
    (width, height*3/3),
    (0, height*3/3)])


    dst = np.float32([(0, 0),
    (width, 0),
    (width, height/3),
    (0, height/3)])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    binary_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)



    return binary_warped

focal = 0
""" Wrapper of Rotating a Image """
def rotate_along_axis_inv(img, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
    global focal
    
    # Get radius of rotation along 3 axes
    rtheta, rphi, rgamma = get_rad(theta, phi, gamma)
    
    
    d = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    dz = focal

    # Get projection matrix
    mat = get_M(img, rtheta, rphi, rgamma, dx, dy, dz)

    T = np.array([  [1, 0, 0],
                    [0, 1, -700],
                    [0, 0, 1]])

    M = np.dot(mat,T)

    M_inv = np.linalg.inv(M)
    
    return cv2.warpPerspective(img, M_inv, (img.shape[1], img.shape[0]*3), flags=cv2.INTER_LINEAR)


def rotate_along_axis(img, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
    global focal
    
    # Get radius of rotation along 3 axes
    rtheta, rphi, rgamma = get_rad(theta, phi, gamma)
    
    # Get ideal focal length on z axis
    # Change this section to other axis if needed
    d = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    dz = focal

    # Get projection matrix
    mat = get_M(img, rtheta, rphi, rgamma, dx, dy, dz)
    
    return cv2.warpPerspective(img, mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


""" Get Perspective Projection Matrix """
def get_M(img, theta, phi, gamma, dx, dy, dz):
    global focal
    
    w = img.shape[1]
    h = img.shape[0]
    f = focal

    # Projection 2D -> 3D matrix
    A1 = np.array([ [1, 0, -w/2],
                    [0, 1, -h/2],
                    [0, 0, 1],
                    [0, 0, 1]])
    
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])
    
    RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
    
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([  [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([ [f, 0, w/2, 0],
                    [0, f, h/2, 0],
                    [0, 0, 1, 0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))


def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_smoothing(image, kernel_size=7):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detect_edges(image, low_threshold=150, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)


def histogram(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    return histogram

def filter_region(image, vertices):
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])
    return cv2.bitwise_and(image, mask)


def select_region(image):
    bl_x=0
    bl_y=0.9
    br_x=0.15
    br_y=1
    tl_x=0.3
    tl_y=0.65
    tr_x=0.7
    tr_y=0.65


    rows, cols = image.shape[:2]
    bottom_left = [cols*bl_x, rows*bl_y]
    top_left = [cols*tl_x, rows*tl_y]
    bottom_right= [cols*br_x, rows*br_y]
    top_right = [cols*tr_x, rows*tr_y]

    bottom_left1 = [cols * (1-bl_x), rows * bl_y]
    top_left1 = [cols * (1-tl_x), rows * tl_y]
    bottom_right1 = [cols * (1-br_x), rows * br_y]
    top_right1 = [cols * (1-tr_x), rows * tr_y]


    vertices_l = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    vertices_r = np.array([[bottom_left1, top_left1, top_right1, bottom_right1]], dtype=np.int32)
    a=filter_region(image, vertices_l)
    b=filter_region(image, vertices_r)
    result=a+b
    return result


def hough_lines(image):
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=60, minLineLength=60, maxLineGap=200)


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros(
    (
    img.shape[0],
    img.shape[1],
    3
    ),
    dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    return img


def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    
    white_mask = cv2.inRange(converted, lower, upper)

    # yellow color mask
    lower = np.uint8([10, 0, 100])
    upper = np.uint8([255, 255, 255])
    # lower = np.uint8([40,20,100])
    # upper = np.uint8([255,255,255])
    yellow_mask = cv2.inRange(converted, lower, upper)

    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    return cv2.bitwise_and(image, image, mask=mask)

prev_poly_left = []
prev_poly_right = []

def lane_mid(image, lines):
    global prev_poly_left, prev_poly_right
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    inaccuracy_count=0
    reliability=0
    going_pixels=[0,0]
    y_pixel=400
    img = np.copy(image)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        if (x2 - x1) == 0:
            slope = 100
        else:
            slope = float(y2 - y1) / float(x2 - x1)

        if abs(math.fabs(slope)) < 0.4:
            inaccuracy_count+=1
            continue
        if ((x1 >= float(image.shape[1] / 2)) ):#and slope>=0) :
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])
        elif ((x1 < float(image.shape[1] / 2)) ):#and 0>=slope):
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])

    ########## left max, right min##############
    #print(inaccuracy_count ,len(lines))
    min_y = img.shape[0] * 0.4
    max_y = img.shape[0]

    # print('left_line_x: ', left_line_x, 'left_line_y: ', left_line_y)
    # print('right_line_x: ', right_line_x, 'right_line_y: ', right_line_y)
    if (inaccuracy_count>=5 or len(lines)<=2):
        going_pixels=[0,0]
        reliability=0

    elif (len(left_line_x)!=0 and len(right_line_x)!=0):
        
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))


        prev_poly_left.append(poly_left)
        if (len(prev_poly_left) > 5):
            prev_poly_left.remove(prev_poly_left[0])

        poly_left = (sum(prev_poly_left) / len(prev_poly_left))

        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))

        prev_poly_right.append(poly_right)
        if (len(prev_poly_right) > 5):
            prev_poly_right.remove(prev_poly_right[0])

        poly_right = (sum(prev_poly_right) / len(prev_poly_right))


        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))


        img = draw_lines(
            img,
            [[
                [left_x_start, max_y, left_x_end, int(min_y)],
                [right_x_start, max_y, right_x_end, int(min_y)],
                ]],
                [0,0,255],
                3
            )


        left_x=poly_left[1]*y_pixel+poly_left[0]
        right_x=poly_right[1]*y_pixel+poly_right[0]
        going_pixels = [(left_x+right_x)/2,y_pixel]
        reliability=1

    elif(len(right_line_x)!=0):
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))

        prev_poly_right.append(poly_right)
        if (len(prev_poly_right) > 5):
            prev_poly_right.remove(prev_poly_right[0])

        poly_right = (sum(prev_poly_right) / len(prev_poly_right))

        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))



        img = draw_lines(
            img,
            [[
                [right_x_start, max_y, right_x_end, int(min_y)],
                ]],
                [0,0,255],
                3
            )
        
        while len(prev_poly_left) > 0 : prev_poly_left.pop()
        right_x=poly_right[1]*y_pixel+poly_right[0]
        going_pixels = [(right_x-280),y_pixel]
        reliability=1

    elif (len(left_line_x)!=0):
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))

        prev_poly_left.append(poly_left)
        if (len(prev_poly_left) > 5):
            prev_poly_left.remove(prev_poly_left[0])

        poly_left = (sum(prev_poly_left) / len(prev_poly_left))

        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

        

        img = draw_lines(
            img,
            [[
                [left_x_start, max_y, left_x_end, int(min_y)],
                ]],
                [0,0,255],
                3
            )
        while len(prev_poly_right) > 0 : prev_poly_right.pop()
        left_x=poly_left[1]*y_pixel+poly_left[0]
        going_pixels = [(left_x+280),y_pixel]
        reliability=1


    #return img, going_pixels
    return img, going_pixels, reliability



def find_delta(image_shape, going_pixels, line_image):

    ############################# angle1 ###################################
    #print(going_pixels[0]-(image_shape[1]/2))
    angle_radian = math.atan(((going_pixels[0]-(image_shape[1]/2))/2)/(image_shape[0]-going_pixels[1])) ## /2
    angle_degree = angle_radian * (180/np.pi)

    #print('mid x1', mid_x1, 'mid x2', mid_x2, 'y1', y1, 'y2', y2)
    #print('radian1: ',angle_radian, 'degree1', angle_degree)

    ############################# pid1 ###################################

    fontType = cv2.FONT_HERSHEY_SIMPLEX
    p = 0.13


    degree = "left" if angle_degree < 0 else "right"
    degree_text = str(round(abs(angle_degree), 3)) + '. ' + degree
    cv2.putText(line_image, degree_text, (30, 100), fontType, 1., (255, 255, 255), 3)

    ########################### distance ######################################
    direction = "left" if going_pixels[0]-(image_shape[1]/2) < 0 else "right"
    deviation_text = 'target is ' + str(round(abs(going_pixels[0]-(image_shape[1]/2)), 3)) + 'pixel ' + direction + ' of center'
    cv2.putText(line_image, deviation_text, (30, 150), fontType, 1., (255, 255, 255), 3)

    ########################## FPS ####################################################

    # deviation_text = str(round(fps, 3)) + ' FPS'
    # cv2.putText(line_image, deviation_text, (30, 50), fontType, 1., (255, 255, 255), 3)

    return line_image, -angle_degree*p


#############################

def _create_config(height, width):

    center_x = 670
    center_y = height / 2
    x_top_factor = 0.04
    x_lower_factor = 0.5
    lower_left = [center_x - x_lower_factor * width, height]
    lower_right = [center_x + x_lower_factor * width, height]
    top_left = [center_x - x_top_factor * width, center_y + height / 10]
    top_right = [center_x + x_top_factor * width, center_y + height / 10]

    roi_matrix = np.int32([
    [0,height],[0,0],[width,0],[width,height]
    ])


    src = np.float32([(200, height/2.0 + 20),
    (width - 200, height/2.0 + 20),
    (0 - 100, height-100),
    (width + 100, height-100)])


    dst = np.float32([(0, 0),
    (width, 0),
    (0, height),
    (width, height)])

   
    return src, dst, roi_matrix

class TrackConsumer:
    """
    A class to encapsulate the lane departure consumer, this class will subscribe
    to the `lanes_video` topic to get frames from the lanes camera input
    than it will apply the LaneDepartureDetector on that input to get the
    processed frame and the actual departure in meters
    """
    def __init__(self):
        rospy.init_node('ros_track_consumer')
        self.bridge = CvBridge()
        self.config = None
        self.detector = None
        self.image_subscription = rospy.Subscriber("lane_images", Image, self.callback)
        self.departure_publisher = rospy.Publisher("image_lane_departure", String, queue_size=100)
        self.path_pub = rospy.Publisher("lane_path", Path, queue_size = 100)
        self.image_publisher = rospy.Publisher("image_lane_detector", Image, queue_size=100)
        self.control_publisher = rospy.Publisher("/Lane_ack_vel", AckermannDriveStamped, queue_size=100)


    def callback(self, data):
        try:
            # Read the image from the input topic:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError, e:
            print e


        kernel = np.ones((3, 3), np.uint8)


        cv_image = cv2.erode(cv_image, kernel, iterations=1)

        cv_image = cv2.dilate(cv_image, kernel, iterations=1)

        kernel2 = np.zeros((3, 3), np.int32)
        kernel2[0,0] = -1
        kernel2[1,0] = -1
        kernel2[2,0] = -1
        kernel2[0,2] = 1
        kernel2[1,2] = 1
        kernel2[2,2] = 1

        

        frame = np.zeros((cv_image.shape[0],cv_image.shape[1],3),np.uint8)


        frame[:,:,0] = cv_image
        frame[:,:,1] = cv_image
        frame[:,:,2] = cv_image

        image_shape=[frame.shape[0],frame.shape[1]]


        
        frame2 = rotate_along_axis_inv(frame, theta=93+180, dy = 20)
        frame2 = cv2.erode(frame2, kernel, iterations=3)

       

        departure = 10

        ackermann_cmd = AckermannDriveStamped()

        blurred_image = apply_smoothing(frame2)

        edge_image = detect_edges(frame2)

        # masked_image = select_region(edge_image)

        # Perspective_lines = WarpPerspecitve(edge_image)

        hough_line = hough_lines(edge_image)

        #print(type(hough_line))
        if(type(hough_line).__module__ == np.__name__):
            lane_detection, deviation_pixels, reliability = lane_mid(frame2, hough_line)

            # list_of_lines = hough_lines(Perspective_lines)

            #frame_warped = WarpPerspecitve(frame)

            #line_image = draw_lines(frame_warped, list_of_lines)
            if(reliability==1):
                final_image, image_delta=find_delta(image_shape, deviation_pixels, lane_detection)

                cv2.line(final_image, (final_image.shape[1]/2,0), (final_image.shape[1]/2,final_image.shape[0]), (255,0,255), 2)

                if (image_delta>28):
                    image_delta=28
                elif(image_delta<-28):
                    image_delta=-28

                L=1.3
                ld=2.5
                final_image_theta=math.atan(2*L*math.sin(image_delta*np.pi/180)/ld)*(180/np.pi)
                ackermann_cmd.drive.steering_angle = final_image_theta * math.pi / 180
                ackermann_cmd.drive.speed = 2.0

 
                self.control_publisher.publish(ackermann_cmd)
                # cv2.imshow('frame2', final_image)
                try:
                    self.image_publisher.publish(self.bridge.cv2_to_imgmsg(final_image))
                except CvBridgeError as e:
                    print(e)
            else:

                departure = departure - 5

                cv2.line(lane_detection, (lane_detection.shape[1]/2,0), (lane_detection.shape[1]/2,lane_detection.shape[0]), (255,0,255), 2)
                # cv2.imshow('frame2', lane_detection)
                try:
                    self.image_publisher.publish(self.bridge.cv2_to_imgmsg(lane_detection))
                except CvBridgeError as e:
                    print(e)


        else:
            # cv2.imshow('frame2', frame)
            departure = departure - 10
            cv2.line(frame2, (frame2.shape[1]/2,0), (frame2.shape[1]/2,frame2.shape[0]), (255,0,255), 2)
            try:
                self.image_publisher.publish(self.bridge.cv2_to_imgmsg(frame2))
            except CvBridgeError as e:
                print(e)


    

        

        self.departure_publisher.publish(str(departure))


def main(args):
    # Init the consumer:
    consumer = TrackConsumer()
    # Continue to spin
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down vison node."


if __name__ == '__main__':
    main(sys.argv)