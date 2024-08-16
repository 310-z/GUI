import cv2
import numpy as np
import math
import os

def bizhi(w,h,distance):
    R = np.round(w/distance)
    distance_h = np.round(h/R)
    distance_w = distance
    distance_h = distance_h  + 15
    return distance_w,distance_h

def draw_reange_RSSCap(x1,y1,distance_h,angle):
    end_x = int(x1 - distance_h * np.cos(np.radians(angle)))
    end_y = int(y1 - distance_h * np.sin(np.radians(angle)))
    return end_x,end_y
def points_RSSCap(x1,y1,x2,y2):
    x3    = x2 + (x2-x1)
    y3    = y2 - (y1-y2)
    return x1,y1,x3,y3
def regis_RSSCap(ultra_image,mri_image):
    image = cv2.resize(mri_image, (512, 512))
    cricles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=15, param2=10, minRadius=0,
                               maxRadius=0)
    cricles = np.uint16(np.around(cricles))
    ori = []
    for i in cricles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (255, 255, 0), 2)
        ori.append((i[0], i[1]))
    ori.sort()
    (x1, y1) = ori[0]
    (x2, y2) = ori[1]
    x1,y1,x2,y2 = points_RSSCap(x1,y1,x2,y2)
    x = abs(int(x1) - int(x2))
    y = abs(int(y1) - int(y2))
    distance = math.sqrt(x ** 2 + y ** 2)
    uh, uw, c = ultra_image.shape
    distance_w, distance_h = bizhi(uw, uh, distance)
    tanx = abs(int(x2) - int(x1)) / abs(int(y2) - int(y1))
    hudu = math.atan(tanx)
    angle =  hudu * 180 / math.pi
    end_x, end_y = draw_reange_RSSCap(x1, y1, distance_h, angle)
    end_x_1, end_y_1 = draw_reange_RSSCap(x2, y2, distance_h, angle)
    # cv2.line(image_1, (x1, y1), (end_x, end_y), (255, 255, 255), 2)
    # cv2.line(image_1, (x2, y2), (end_x_1, end_y_1), (255, 255, 255), 2)
    # cv2.line(image_1, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # cv2.line(image_1, (end_x, end_y), (end_x_1, end_y_1), (255, 255, 255), 2)
    m1 = (x1, y1)
    m2 = (end_x, end_y)
    m3 = (x2, y2)
    m4 = (end_x_1, end_y_1)
    return image,  m1, m2, m3, m4