import cv2
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import os


def smooth_image_edges(image):
    kernel_size = (7, 7)
    blurred_image = cv2.blur(image, kernel_size)
    return blurred_image

def add_black_border(image):
    height, width = image.shape[:2]
    # 计算新的高度和宽度，考虑黑边的厚度
    new_height = height +100
    new_width = width + 100
    # 创建一个新的全黑图像，大小为新的高度和宽度
    black_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    # 将原始图像复制到新的图像中，位置根据黑边的厚度调整
    black_image[50:50 + height, 50:50 + width] = image
    return black_image


def Denoising(img):
    kernel = np.ones((15, 15), np.uint8)

    # 图像开运算
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    result = cv2.morphologyEx(result,cv2.MORPH_CLOSE,kernel)
    return result


def SCM(x):
    y1 = 3.107e-11 * x * x * x * x * x - 3.888e-08 * x * x * x * x + 1.71e-05 * x * x * x - 0.002251 * x * x - 0.2075 * x - 51.57
    y2 = -4.669e-12 * x * x * x * x * x + 8.402e-09 * x * x * x * x - 5.95e-06 * x * x * x + 0.002151 * x * x - 0.3171 * x - 26.96
    return y1,y2

def SSCap(x):
    y1 = 2.237e-11 * x * x * x * x * x - 2.559e-08 * x * x * x * x + 1.121e-05 * x * x * x - 0.001755 * x * x + 0.222075 * x -24.48
    y2 = -2.1458e-11 * x * x * x * x * x + 2.5795e-08 * x * x * x * x - 1.0195e-05 * x * x * x + 0.0017795 * x * x + 0.109505 * x - 19.08
    return y1,y2

def correct_LSCM(image_path,input,save_path):
            img = cv2.imread(image_path)
            img = img[0:783,275:1000]
            #img = add_black_border(img)
            img1 = cv2.resize(img, (512,512))
            img1  = Denoising(img1)
            img1  = smooth_image_edges(img1)
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            re, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite("./test.png", th)
            img_LSCM = cv2.imread("./test.png")
            h, w, c = img_LSCM.shape
            new_img = np.zeros((h, w, c))
            for j in range(512):
                m = []
                for i in range(512):
                    if all(img_LSCM[i, j] == [255, 255, 255]):
                        m.append(i)

                if len(m) == 0:
                    for i in range(512):
                        new_img[i, j] = [0, 0, 0]
                else:
                    i_min = m[0]
                    i_max = m[-1]
                    y1, y2 = SCM(j)
                    i_min_new = i_min +  y1
                    i_max_new = i_max +  y2
                    for i in range(512):
                        if i >= i_min_new and i <= i_max_new:
                            new_img[i, j] = [0, 255, 0]
            new_path  = os.path.join(save_path, input)
            new_img  = cv2.resize(new_img , (512, 512))
            print(new_path)
            cv2.imwrite(new_path,new_img)

def correct_LSSCap(image_path,input,save_path):
            img = cv2.imread(image_path)
            img = img[0:783, 275:1000]
           # img = add_black_border(img)
            img1 = cv2.resize(img, (512,512))
            img1  = Denoising(img1)
            img1  = smooth_image_edges(img1)
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            re, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite("./test.png", th)
            img_LSCM = cv2.imread("./test.png")
            h, w, c = img_LSCM.shape
            new_img = np.zeros((h, w, c))
            for j in range(512):
                m = []
                for i in range(512):
                    if all(img_LSCM[i, j] == [255, 255, 255]):
                        m.append(i)
                if len(m) == 0:
                    for i in range(512):
                        new_img[i, j] = [0, 0, 0]
                else:
                    i_min = m[0]
                    i_max = m[-1]
                    y1, y2 = SSCap(j)
                    i_min_new = i_min + y1
                    i_max_new = i_max + y2
                    for i in range(512):
                        if i >= i_min_new and i <= i_max_new:
                            new_img[i, j] = [255,0,255]
            new_path = os.path.join(save_path, input)
            new_img = cv2.resize(new_img, (512, 512))
            print(new_path)
            cv2.imwrite(new_path, new_img)

def correct_RSCM(image_path,input,save_path):
            img = cv2.imread(image_path)
            img = img[0:783, 275:1000]
          #  img = add_black_border(img)
            img1 = cv2.resize(img, (512,512))
            img1  = Denoising(img1)
            img1  = smooth_image_edges(img1)
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            re, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite("./test.png", th)
            img_LSCM = cv2.imread("./test.png")
            h, w, c = img_LSCM.shape
            new_img = np.zeros((h, w, c))
            for j in range(512):
                m = []
                for i in range(512):
                    if all(img_LSCM[i, j] == [255, 255, 255]):
                        m.append(i)
                if len(m) == 0:
                    for i in range(512):
                        new_img[i, j] = [0, 0, 0]
                else:
                    i_min = m[0]
                    i_max = m[-1]
                    y1, y2 = SCM(j)
                    i_min_new = i_min + y1
                    i_max_new = i_max + y2
                    for i in range(512):
                        if i >= i_min_new and i <= i_max_new:
                            new_img[i, j] = [0,0,255]
            new_path = os.path.join(save_path, input)
            new_img = cv2.resize(new_img, (512, 512))
            print(new_path)
            cv2.imwrite(new_path, new_img)

def correct_RSSCap(image_path,input,save_path):
            img = cv2.imread(image_path)
            img = img[0:783, 275:1000]
          #  img = add_black_border(img)
            img1 = cv2.resize(img, (512,512))
            img1  = Denoising(img1)
            img1  = smooth_image_edges(img1)
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            re, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite("./test.png", th)
            img_LSCM = cv2.imread("./test.png")
            h, w, c = img_LSCM.shape
            new_img = np.zeros((h, w, c))
            for j in range(512):
                m = []
                for i in range(512):
                    if all(img_LSCM[i, j] == [255, 255, 255]):
                        m.append(i)
                if len(m) == 0:
                    for i in range(512):
                        new_img[i, j] = [0, 0, 0]
                else:
                    i_min = m[0]
                    i_max = m[-1]
                    y1, y2 = SSCap(j)
                    i_min_new = i_min + y1
                    i_max_new = i_max + y2
                    for i in range(512):
                        if i >= i_min_new and i <= i_max_new:
                            new_img[i, j] = [255,0,0]

            new_path = os.path.join(save_path, input)
            new_img = cv2.resize(new_img, (512, 512))
            print(new_path)
            cv2.imwrite(new_path, new_img)


