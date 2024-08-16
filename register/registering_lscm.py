import cv2
import numpy as np
import math
import os


def diejia(image,img_ultra,m1,m2,m3,m4):
    image = cv2.resize(image, (512, 512))
    hh,ww,c = image.shape
    # 定义超声图像的尺寸
    h, w, c = img_ultra.shape
    # 定义超声图像上的四个角点坐标（左上、右上、左下、右下）
    src_points = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    dst_points = np.float32([m1, m2, m3, m4])
    cv2.line(image, m1, m2, (255, 255, 255), 3)
    cv2.line(image, m1, m3, (255, 255, 255), 3)
    cv2.line(image, m2, m4, (255, 255, 255), 3)
    cv2.line(image, m3, m4, (255, 255, 255), 3)
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # 应用透视变换
    result = cv2.warpPerspective(img_ultra, perspective_matrix, (hh, ww))
    re = cv2.add(image, result)
    return re,result,image

#计算超声图像和核磁对应肌肉的尺寸比值
def bizhi(w,h,distance):
    R = np.round(w/distance)
    distance_h = np.round(h/R)
    distance_w = distance
    distance_h = distance_h  + 15
    return distance_w,distance_h

def draw_reange_LSCM(x1,y1,distance_h,angle):
    end_x = int(x1 + distance_h * np.cos(np.radians(angle)))
    end_y = int(y1 + distance_h * np.sin(np.radians(angle)))
    return end_x,end_y
def points_LSCM(x1,y1,x2,y2):
    x3    = x2 - (x1-x2)
    y3    = y2 + (y2-y1)
    return x1,y1,x3,y3

def regis_LSCM(ultra_image,mri_image):
    image     = cv2.resize(mri_image,(512,512))
    cricles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=15, param2=10, minRadius=0, maxRadius=0)

    cricles = np.uint16(np.around(cricles))
    ori = []
    for i in cricles[0, :]:
        cv2.circle(image,(i[0],i[1]),i[2],(255,255,0),2)
        ori.append((i[0],i[1]))

    ori.sort()
    (x1, y1) = ori[1]
    (x2, y2) = ori[0]
    x1,y1,x2,y2 = points_LSCM(x1,y1,x2,y2)
    x = abs(int(x1) - int(x2))
    y = abs(int(y1) - int(y2))
    distance = math.sqrt(x ** 2 + y ** 2)
    uh, uw,c = ultra_image.shape
    distance_w, distance_h = bizhi(uw, uh, distance)
    tanx = abs(int(x2) - int(x1)) / abs(int(y2) - int(y1))
    hudu = math.atan(tanx)
    angle = hudu * 180 / math.pi
    end_x, end_y= draw_reange_LSCM(x1, y1, distance_h, angle)
    end_x_1, end_y_1= draw_reange_LSCM(x2, y2, distance_h, angle)
    m1   =  (x1, y1)
    m2   = (end_x, end_y)
    m3   = (x2,y2)
    m4   = (end_x_1, end_y_1)
    return image,m1,m2,m3,m4

def diejia_LSCM(image,img_1,m1,m2,m3,m4):
    #image   = cv2.imread(image_path)
    image   = cv2.resize(image,(512,512))
    hh,ww,c = image.shape
    image_LSCM = np.zeros((hh, ww, c))
    # 定义超声图像的尺寸
    area0 =0
    for i in range(hh):
        for j in range(ww):
            if np.all(image[i][j]==[0,128,0]):  #LSCM
                image_LSCM[i][j] = [255,255,255]
                area0 = area0+1
    h, w, c = img_1.shape
    # 定义超声图像上的四个角点坐标（左上、右上、左下、右下）
    img_1 = np.zeros((h, w), dtype=np.uint8)

    # 将图像转换成BGR格式并添加alpha通道
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR)
    img_1[:] = (255, 255, 255)  # 在每个通道上都设置值为255表示白色
    src_points = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    dst_points = np.float32([m1, m2, m3, m4])
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    # 应用透视变换
    result = cv2.warpPerspective(img_1, perspective_matrix, (hh, ww))
    cv2.imwrite("./test.png",image_LSCM)
    image_2 = cv2.imread("./test.png")
    re = cv2.add(image_2, result)
    h1,w1,c = re.shape
    area1  = 0
    for i in range(h1):
        for j in range(w1):
            if np.all(re[i][j]==[255,255,255]):  #LSCM
                area1  = area1+ 1
    h2, w2, c = result.shape
    area2 = 0
    for i in range(h2):
        for j in range(w2):
            if np.all(result[i][j] == [255, 255, 255]):  # LSCM
                area2 = area2 + 1
    area = area1 - area2
    area = area0 - area
    return area