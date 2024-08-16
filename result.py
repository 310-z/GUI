import sys
from nibabel import Nifti1Image
from PyQt5 import QtGui,QtWidgets
import copy
from mayavi import mlab
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import nibabel as nib
from register.registering_lscm import regis_LSCM,diejia
from register.registering_rscm import regis_RSCM
from register.registering_lsscap import regis_LSSCap
from register.registering_rsscap import regis_RSSCap
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from Jingunet_re.GinGuNet.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image, show_config
import os
from tqdm import tqdm
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from register.rectify import correct_LSCM,correct_RSCM,correct_LSSCap,correct_RSSCap
import shutil
import time

class WorkerThread(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self,file_path,save_path):
        super().__init__()
        print(file_path)
        self.file_path = file_path  # 将file_path作为属性保存
        self.save_path = save_path  # 将file_path作为属性保存
        self._stop_requested = False

    def run(self):

        def makeDir(path):
            if not os.path.exists(path):
                if not os.path.isfile(path):
                    os.makedirs(path)

        save_path = self.save_path
        makeDir(save_path)
        mp4 = cv2.VideoCapture(self.file_path)  # 读取视频
        is_opened = mp4.isOpened()
        sumps = mp4.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧数print(fps,sumps,"帧")
        sumps = int(sumps)
        if sumps > 100:
              s     = int(sumps) - 100
              n = sumps
              i = 2
              progress = 0
              while is_opened:
                  (flag, frame) = mp4.read()  # 读取图片
                  file_name = str(i - 1) + ".png"
                  if flag == True:
                      cv2.imwrite(save_path + "/" + file_name, frame)
                      if i >= n:  # 截断等于帧总数
                          break
                      else:
                          i = i + 1
                      print(save_path + "/" + file_name)
                      progress = (progress + 1)  # 循环使用0-100的数值
                      if progress > s:
                          self.progress_signal.emit(int(progress-s+2))
        else:
            s       = 100 - int(sumps)
            n = sumps
            i = 2
            progress = s
            while is_opened:
                (flag, frame) = mp4.read()  # 读取图片
                file_name = str(i - 1) + ".png"
                if flag == True:
                    cv2.imwrite(save_path + "/" + file_name, frame)
                    if i >= n:  # 截断等于帧总数
                        break
                    else:
                        i = i + 1
                    print(save_path + "/" + file_name)
                    progress = (progress + 1)   # 循环使用0-100的数值
                    self.progress_signal.emit(int(progress+s))
        # self.progress_signal.emit(0)
        # self.finished.emit()


class WorkerThread_seg(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, name):
        super().__init__()
        self.name = name  # 将file_path作为属性保存
        self._stop_requested = False

    def run(self):
        def generate(model_path, num_classes):
            net = DeepLab(num_classes=num_classes)
            net.load_state_dict(torch.load(model_path))
            net = net.eval()
            net = nn.DataParallel(net)
            net = net.cuda()
            return net

        def detect_image(image, model_path, num_classes):
            net = generate(model_path, num_classes)
            mix_type = 1
            image = cvtColor(image)
            old_img = copy.deepcopy(image)
            orininal_h = np.array(image).shape[0]
            orininal_w = np.array(image).shape[1]
            image_data, nw, nh = resize_image(image, (512, 512))
            image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
            with torch.no_grad():
                images = torch.from_numpy(image_data)
                images = images.cuda()
                pr = net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
                pr = pr[int((512 - nh) // 2): int((512 - nh) // 2 + nh), \
                     int((512 - nw) // 2): int((512 - nw) // 2 + nw)]
                pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
                pr = pr.argmax(axis=-1)

            if num_classes <= 21:
                colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
                          (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                          (192, 0, 128),
                          (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                          (0, 64, 128),
                          (128, 64, 12)]
            if mix_type == 0:
                seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
                image = Image.fromarray(np.uint8(seg_img))
                image = Image.blend(old_img, image, 0.7)

            elif mix_type == 1:
                seg_img = np.reshape(np.array(colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
                image = Image.fromarray(np.uint8(seg_img))

            elif mix_type == 2:
                seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
                image = Image.fromarray(np.uint8(seg_img))

            return image

        def makeDir(path):
            if not os.path.exists(path):
                if not os.path.isfile(path):
                    os.makedirs(path)

        self.progress_signal.emit(0)
        name = self.name
        makeDir("img")
        makeDir("./img"+"/"+str(name))
        dir_origin_path = "./picture"+"/"+str(name)
        dir_save_path = "./img"+"/"+str(name)
        model_path = "./weights/" + name + ".pth"
        num_classes = 2
        img_names = os.listdir(dir_origin_path)
        n         = len(img_names)
        if n >= 100:
            s = n - 100
            progress  = 0
            for i in range(1, n + 1):
                image_path = os.path.join(dir_origin_path, str(i) + ".png")
                print(image_path)
                image = Image.open(image_path)
                r_image = detect_image(image, model_path, num_classes)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, str(i) + ".png"))
                progress = (progress + 1) % n  # 循环使用0-100的数值
                if progress > s:
                    self.progress_signal.emit(int(progress-s+2))
        else:
            s = 100 - n
            progress = 0
            for j in range(int(s)+1):
                 progress = j
                 self.progress_signal.emit(int(progress))
            for i in range(1, n + 1):
                print(i)
                image_path = os.path.join(dir_origin_path, str(i) + ".png")
                print(image_path)
                image = Image.open(image_path)
                r_image = detect_image(image, model_path, num_classes)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, str(i) + ".png"))
                progress = progress + 1  # 循环使用0-100的数值
                print(progress)
                self.progress_signal.emit(int(progress))
       #progress = 0  # 循环使用0-100的数值
        # self.progress_signal.emit(0)
        # self.finished.emit()
        src_path = r'./img/' + str(name) + "/"
        sav_path = "./img/" + str(name) + ".mp4"
        all_files = os.listdir(src_path)
        files = []
        for s in all_files:
            na = s[0:-4]
            na = int(na)
            files.append(na)
        files.sort()
        all_files = []
        for i in files:
            nas = str(i) + ".png"
            all_files.append(nas)
        index = len(all_files)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
        # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
        size = (512, 512)
        videowrite = cv2.VideoWriter(sav_path, fourcc, 20, size)  # 2是每秒的帧数，size是图片尺寸
        # 5.临时存放图片的数组
        img_array = []

        # 6.读取所有jpg格式的图片 (这里图片命名是0-index.jpg example: 0.jpg 1.jpg ...)
        for filename in [src_path + r'{0}.png'.format(i) for i in range(1, index + 1)]:
            print(filename)
            img = cv2.imread(filename)
            if img is None:
                print(filename + " is error!")
                continue
            img_array.append(img)
        # 7.合成视频
        for i in range(1, index + 1):
            img_array[i - 1] = cv2.resize(img_array[i - 1], (512, 512))
            videowrite.write(img_array[i - 1])
            print('第{}张图片合成成功'.format(i))
class WorkerThread_cor(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, name):
        super().__init__()
        self.name = name  # 将file_path作为属性保存
        self._stop_requested = False

    def run(self):
        def makeDir(path):
            if not os.path.exists(path):
                if not os.path.isfile(path):
                    os.makedirs(path)

        self.progress_signal.emit(0)
        name = self.name
        makeDir("./correct")
        makeDir("./correct" + "/" + str(name))
        input_path = "./img" + "/" + str(name)
        save_path = "./correct" + "/" + str(name)
        name = self.name
        inputs = os.listdir(input_path)
        n      = len(inputs)
        if n >= 100:
            s = n - 100
            progress = 0
            for input in inputs:
                image_path = os.path.join(input_path, input)
                if str(name) == "LSCM":
                    correct_LSCM(image_path, input, save_path)
                elif str(name) == "LSSCap":
                    correct_LSSCap(image_path, input, save_path)
                elif str(name) == "RSCM":
                    correct_RSCM(image_path, input, save_path)
                elif str(name) == "RSSCap":
                    correct_RSSCap(image_path, input, save_path)
                progress = (progress + 1)   # 循环使用0-100的数值
                self.progress_signal.emit(int(progress-s+2))
        else :
            s = 100 - n
            progress = 0
            for j in range(int(s)+1):
                progress = progress + 1
                self.progress_signal.emit(int(progress))
            for input in inputs:
                image_path = os.path.join(input_path, input)
                if str(name) == "LSCM":
                    correct_LSCM(image_path, input, save_path)
                elif str(name) == "LSSCap":
                    correct_LSSCap(image_path, input, save_path)
                elif str(name) == "RSCM":
                    correct_RSCM(image_path, input, save_path)
                elif str(name) == "RSSCap":
                    correct_RSSCap(image_path, input, save_path)
                progress = (progress + 1)  # 循环使用0-100的数值
                self.progress_signal.emit(int(progress))

        src_path = r'./correct/' + str(name) +"/"
        sav_path = "./correct/"+str(name)+".mp4"
        all_files = os.listdir(src_path)
        files = []
        for s in all_files:
            na = s[0:-4]
            na = int(na)
            files.append(na)
        files.sort()
        all_files = []
        for i in files:
            nas = str(i) + ".png"
            all_files.append(nas)
        index = len(all_files)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
        # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
        size = (512, 512)
        videowrite = cv2.VideoWriter(sav_path, fourcc, 20, size)  # 2是每秒的帧数，size是图片尺寸
        # 5.临时存放图片的数组
        img_array = []

        # 6.读取所有jpg格式的图片 (这里图片命名是0-index.jpg example: 0.jpg 1.jpg ...)
        for filename in [src_path + r'{0}.png'.format(i) for i in range(1, index+1)]:
            img = cv2.imread(filename)
            if img is None:
                # print(filename + " is error!")
                continue
            img_array.append(img)
        # 7.合成视频
        for i in range(1, index+1):
            img_array[i-1] = cv2.resize(img_array[i-1], (512, 512))
            videowrite.write(img_array[i-1])
            print('第{}张图片合成成功'.format(i))

        # self.progress_signal.emit(0)
        # self.finished.emit()
class WorkerThread_reg(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, name,image_path1,image_path2):
        super().__init__()
        self.name = name  # 将file_path作为属性保存
        self.image_path1 = image_path1
        self.image_path2 = image_path2
        self._stop_requested = False

    def run(self):
        def makeDir(path):
            if not os.path.exists(path):
                if not os.path.isfile(path):
                    os.makedirs(path)

        def together(image1, image2):
            image1 = cv2.resize(image1, (512, 512))
            image2 = cv2.resize(image2, (512, 512))
            img = cv2.subtract(image1, image2)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            re, th = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
            return th

        self.progress_signal.emit(0)
        name = self.name
        makeDir("./registering")
        makeDir("./registering" + "/" + str(name))
        input_path = "./correct" + "/" + str(name)
        save_path = "./registering" + "/"+str(name) +"/"
        image_path1 = self.image_path1
        image_path2 = self.image_path2
        image1      = cv2.imread(image_path1)
        image2      = cv2.imread(image_path2)
        img_LSCM    = together(image1,image2)
        img = np.zeros((512,512))
        cv2.imwrite(save_path+"/"+"0.png",img)
        name = self.name
        inputs = os.listdir(input_path)
        n = len(inputs)
        if n >= 100:
            s = n - 100
            progress = 0
            for i in range(n - 2):
                ultra_path = os.path.join(input_path, str(i + 1) + ".png")
                ultra_image = cv2.imread(ultra_path)
                ultra_image = ultra_image[0:783, 275:1000]
                ultra_image = cv2.resize(ultra_image, (512, 512))
                if name == "LSCM":
                     image, m1, m2, m3, m4 = regis_LSCM(ultra_image, img_LSCM)
                elif name == "LSSCap":
                    image, m1, m2, m3, m4 = regis_LSSCap(ultra_image, img_LSCM)
                elif name == "RSCM":
                    image, m1, m2, m3, m4 = regis_RSCM(ultra_image, img_LSCM)
                elif name == "RSSCap":
                    image, m1, m2, m3, m4 = regis_RSSCap(ultra_image, img_LSCM)
                re, result, image = diejia(image1, ultra_image, m1, m2, m3, m4)
                cv2.imwrite(save_path + "/" + str(i + 1) + ".png", result)
                print(save_path + "/" + str(i + 1) + ".png")
                progress = (progress + 1)  # 循环使用0-100的数值
                self.progress_signal.emit(int(progress - s + 2))
        else:
            s = 100 - n
            progress = 0
            for j in range(int(s)+2):
                progress = progress + 1
                self.progress_signal.emit(int(progress))
            for i in range(n - 2):
                ultra_path = os.path.join(input_path, str(i + 1) + ".png")
                ultra_image = cv2.imread(ultra_path)
                ultra_image = ultra_image[0:783, 275:1000]
                ultra_image = cv2.resize(ultra_image, (512, 512))
                if name == "LSCM":
                    image, m1, m2, m3, m4 = regis_LSCM(ultra_image, img_LSCM)
                elif name == "LSSCap":
                    image, m1, m2, m3, m4 = regis_LSSCap(ultra_image, img_LSCM)
                elif name == "RSCM":
                    image, m1, m2, m3, m4 = regis_RSCM(ultra_image, img_LSCM)
                elif name == "RSSCap":
                    image, m1, m2, m3, m4 = regis_RSSCap(ultra_image, img_LSCM)
                re, result, image = diejia(image1, ultra_image, m1, m2, m3, m4)
                cv2.imwrite(save_path + "/" + str(i + 1) + ".png", result)
                print(save_path + "/" + str(i + 1) + ".png")
                progress = (progress + 1)  # 循环使用0-100的数值
                self.progress_signal.emit(int(progress))
        img = np.zeros((512, 512))
        cv2.imwrite(save_path + "/" + str(n - 1) + ".png", img)
        src_path = save_path
        sav_path = "./registering/"  + str(name) + ".mp4"
        all_files = os.listdir(src_path)
        print(all_files)
        files = []
        for s in all_files:
            na = s[0:-4]
            na = int(na)
            files.append(na)
        files.sort()
        print(files)
        all_files = []
        for i in files:
            nas = str(i) + ".png"
            all_files.append(nas)
        print(all_files)
        index = len(all_files)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
        # 完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
        size = (512, 512)
        videowrite = cv2.VideoWriter(sav_path, fourcc, 20, size)  # 2是每秒的帧数，size是图片尺寸
        # 5.临时存放图片的数组
        img_array = []

        # 6.读取所有jpg格式的图片 (这里图片命名是0-index.jpg example: 0.jpg 1.jpg ...)
        for filename in [src_path + r'{0}.png'.format(i) for i in range(0, index)]:
            print(filename)
            img = cv2.imread(filename)
            if img is None:
                print(filename + " is error!")
                continue
            img_array.append(img)
        # 7.合成视频
        for i in range(0, index):
            img_array[i] = cv2.resize(img_array[i], (512, 512))
            videowrite.write(img_array[i])
            print('第{}张图片合成成功'.format(i))
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()


        # 定义视频播放
        self.frame_LSCM = []  # 存图片
        self.detectFlag_LSCM = False  # 检测flag
        self.cap_LSCM = []
        self.timer_camera_LSCM = QTimer()  # 定义定时器

        self.frame_LSSCap = []  # 存图片
        self.detectFlag_LSSCap = False  # 检测flag
        self.cap_LSSCap = []
        self.timer_camera_LSSCap = QTimer()  # 定义定时器

        self.frame_RSCM = []  # 存图片
        self.detectFlag_RSCM = False  # 检测flag
        self.cap_RSCM = []
        self.timer_camera_RSCM = QTimer()  # 定义定时器

        self.frame_RSSCap = []  # 存图片
        self.detectFlag_RSSCap = False  # 检测flag
        self.cap_RSSCap = []
        self.timer_camera_RSSCap = QTimer()  # 定义定时器

        self.button_1.clicked.connect(self.sece_LSCM)
        self.button_1.clicked.connect(self.start_task_LSCM)
        self.button_2.clicked.connect(self.sece_LSSCap)
        self.button_2.clicked.connect(self.start_task_LSSCap)
        self.button_3.clicked.connect(self.sece_RSCM)
        self.button_3.clicked.connect(self.start_task_RSCM)
        self.button_4.clicked.connect(self.sece_RSSCap)
        self.button_4.clicked.connect(self.start_task_RSSCap)
        self.button_MRI_1.clicked.connect(self.sece_LSCM_MRI)
        self.button_MRI_2.clicked.connect(self.sece_LSSCap_MRI)
        self.button_MRI_3.clicked.connect(self.sece_RSCM_MRI)
        self.button_MRI_4.clicked.connect(self.sece_RSSCap_MRI)
        self.button_6.clicked.connect(self.sece_MRI)
        self.start_1.clicked.connect(self.slotStart_LSCM)
        self.stop_1.clicked.connect(self.slotStop_LSCM)
        self.start_2.clicked.connect(self.slotStart_LSSCap)
        self.stop_2.clicked.connect(self.slotStop_LSSCap)
        self.start_3.clicked.connect(self.slotStart_RSCM)
        self.stop_3.clicked.connect(self.slotStop_RSCM)
        self.start_4.clicked.connect(self.slotStart_RSSCap)
        self.stop_4.clicked.connect(self.slotStop_RSSCap)
        self.button_predict.clicked.connect(self.start_task_seg_LSCM)
        self.button_correct.clicked.connect(self.start_task_cor_1)
        self.button_registering.clicked.connect(self.start_task_reg_1)
        self.button_save.clicked.connect(self.visi)
        self.button_5.clicked.connect(self.save)

    def initUI(self):
        self.setWindowTitle('Muscle three-dimensional visualization program')
        self.setGeometry(150, 150, 1511, 830)

        self.progressBar1 = QProgressBar(self)
        self.progressBar1.setRange(0, 100)
        self.progressBar1.setValue(0)
        self.progressBar1.move(507, 252)  # 设置位置，单位为像素

        self.progressBar2 = QProgressBar(self)
        self.progressBar2.setRange(0, 100)
        self.progressBar2.setValue(0)
        self.progressBar2.move(1025, 252)  # 设置位置，单位为像素

        self.progressBar3 = QProgressBar(self)
        self.progressBar3.setRange(0, 100)
        self.progressBar3.setValue(0)
        self.progressBar3.move(507, 544)  # 设置位置，单位为像素

        self.progressBar4 = QProgressBar(self)
        self.progressBar4.setRange(0, 100)
        self.progressBar4.setValue(0)
        self.progressBar4.move(1025,544)  # 设置位置，单位为像素

        self.label = QLabel(self)
        self.label.setFixedSize(1052, 185)  # width height
        self.label.move(460, 0)
        self.label.setStyleSheet("QLabel{background:silver;}")

        self.label_1 = QLabel(self)
        self.label_1.setFixedSize(262, 173)  # width height
        self.label_1.move(507, 282)
        self.label_1.setStyleSheet("QLabel{background:pink;}")

        self.label_MRI_1 = QLabel(self)
        self.label_MRI_1.setFixedSize(180, 174)  # width height
        self.label_MRI_1.move(769, 282)
        self.label_MRI_1.setStyleSheet("QLabel{background:papayawhip;}")

        self.label_2 = QLabel(self)
        self.label_2.setFixedSize(263, 173)  # width height
        self.label_2.move(1024, 282)
        self.label_2.setStyleSheet("QLabel{background:pink;}")

        self.label_MRI_2 = QLabel(self)
        self.label_MRI_2.setFixedSize(180, 174)  # width height
        self.label_MRI_2.move(1287, 282)
        self.label_MRI_2.setStyleSheet("QLabel{background:papayawhip;}")

        self.label_3 = QLabel(self)
        self.label_3.setFixedSize(263, 173)  # width height
        self.label_3.move(507, 574)
        self.label_3.setStyleSheet("QLabel{background:pink;}")

        self.label_MRI_3 = QLabel(self)
        self.label_MRI_3.setFixedSize(180, 174)  # width height
        self.label_MRI_3.move(769, 574)
        self.label_MRI_3.setStyleSheet("QLabel{background:papayawhip;}")

        self.label_4 = QLabel(self)
        self.label_4.setFixedSize(263, 173)  # width height
        self.label_4.move(1024, 574)
        self.label_4.setStyleSheet("QLabel{background:pink;}")

        self.label_MRI_4 = QLabel(self)
        self.label_MRI_4.setFixedSize(180, 174)  # width height
        self.label_MRI_4.move(1287, 574)
        self.label_MRI_4.setStyleSheet("QLabel{background:papayawhip;}")

        self.label_5 = QLabel("Three-dimensional visualization result",self)
        self.label_5.setFixedSize(345, 35)  # width height
        self.label_5.move(77, 72)
        self.label_5.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                    "background-color:lime; color: black;font-size: 20px;")

        self.label_6 = QLabel(self)
        self.label_6.setFixedSize(345, 345)  # width height
        self.label_6.move(77, 108)
        self.label_6.setStyleSheet("QLabel{background:papayawhip;}")

        self.label_7 = QLabel(self)
        self.label_7.setFixedSize(197, 197)  # width height
        self.label_7.move(127, 585)
        self.label_7.setStyleSheet("QLabel{background:powderblue;}")


        self.txt_label_5 = QLabel(self)
        self.txt_label_5.setGeometry(77, 453, 290, 35)
        self.txt_label_5.setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 8)
        font.setBold(True)  # 设置字体加粗
        self.txt_label_5.setFont(font)
        self.txt_label_5.setStyleSheet("QLabel{background:white;}")

        self.button_5 = QPushButton("save", self)
        self.button_5.setFixedSize(60, 35)
        self.button_5.move(367, 453)
        self.button_5.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                        "background-color:yellow; color: black;font-size: 20px;")

        self.button_6 = QPushButton("MRI", self)
        self.button_6.setFixedSize(197, 35)
        self.button_6.move(127, 550)
        self.button_6.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                    "background-color:yellow; color: black;font-size: 20px;")

        self.txt_label = QLabel(self)
        self.txt_label.setGeometry(127, 782, 197, 35)
        self.txt_label.setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 8)
        font.setBold(True)  # 设置字体加粗
        self.txt_label.setFont(font)
        self.txt_label.setStyleSheet("QLabel{background:white;}")
        #定义按钮
        self.button_1 = QPushButton("LSCM", self)
        self.button_1.setFixedSize(140, 35)
        self.button_1.move(630, 247)
        self.button_1.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                         "background-color:moccasin; color: black;font-size: 20px;")

        self.button_MRI_1 = QPushButton("MRI_LSCM", self)
        self.button_MRI_1.setFixedSize(180, 35)
        self.button_MRI_1.move(770, 247)
        self.button_MRI_1.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                    "background-color:yellow; color: black;font-size: 20px;")

        self.button_2 = QPushButton("LSSCap", self)
        self.button_2.setFixedSize(140, 35)
        self.button_2.move(1148, 247)
        self.button_2.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                    "background-color:moccasin; color: black;font-size: 20px;")

        self.button_MRI_2 = QPushButton("MRI_LSSCap", self)
        self.button_MRI_2.setFixedSize(180, 35)
        self.button_MRI_2.move(1288, 247)
        self.button_MRI_2.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                        "background-color:yellow; color: black;font-size: 20px;")

        self.button_3 = QPushButton("RSCM", self)
        self.button_3.setFixedSize(140, 35)
        self.button_3.move(630, 539)
        self.button_3.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                    "background-color:moccasin; color: black;font-size: 20px;")

        self.button_MRI_3 = QPushButton("MRI_RSCM", self)
        self.button_MRI_3.setFixedSize(180, 35)
        self.button_MRI_3.move(770, 539)
        self.button_MRI_3.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                        "background-color:yellow; color: black;font-size: 20px;")

        self.button_4 = QPushButton("RSSCap", self)
        self.button_4.setFixedSize(140, 35)
        self.button_4.move(1148,539)
        self.button_4.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                    "background-color:moccasin; color: black;font-size: 20px;")

        self.button_MRI_4 = QPushButton("MRI_RSSCap", self)
        self.button_MRI_4.setFixedSize(180, 35)
        self.button_MRI_4.move(1288, 539)
        self.button_MRI_4.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                        "background-color:yellow; color: black;font-size: 20px;")

        self.start_1 = QPushButton("start", self)
        self.start_1.setFixedSize(50, 35)
        self.start_1.move(506, 456)
        self.start_1.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                        "background-color:lightgreen; color: black;font-size: 20px;")

        # 定义文本框
        self.txt_label_1 = QLabel(self)
        self.txt_label_1.setGeometry(556, 456, 200, 35)
        self.txt_label_1.setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 8)
        font.setBold(True)  # 设置字体加粗
        self.txt_label_1.setFont(font)
        self.txt_label_1.setStyleSheet("QLabel{background:white;}")

        self.stop_1 = QPushButton("stop", self)
        self.stop_1.setFixedSize(50, 35)
        self.stop_1.move(756, 456)
        self.stop_1.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                   "background-color:lightgreen; color: black;font-size: 20px;")

        self.txt_label_MRI_1 = QLabel(self)
        self.txt_label_MRI_1.setGeometry(806, 456, 142, 35)
        self.txt_label_MRI_1.setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 8)
        font.setBold(True)  # 设置字体加粗
        self.txt_label_MRI_1.setFont(font)
        self.txt_label_MRI_1.setStyleSheet("QLabel{background:white;}")

        self.start_2 = QPushButton("start", self)
        self.start_2.setFixedSize(50, 35)
        self.start_2.move(1024,456)
        self.start_2.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                        "background-color:lightgreen; color: black;font-size: 20px;")

        self.txt_label_2 = QLabel(self)
        self.txt_label_2.setGeometry(1074, 456, 200, 35)
        self.txt_label_2.setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 8)
        font.setBold(True)  # 设置字体加粗
        self.txt_label_2.setFont(font)
        self.txt_label_2.setStyleSheet("QLabel{background:white;}")

        self.stop_2 = QPushButton("stop", self)
        self.stop_2.setFixedSize(50, 35)
        self.stop_2.move(1274,456)
        self.stop_2.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                  "background-color:lightgreen; color: black;font-size: 20px;")

        self.txt_label_MRI_2 = QLabel(self)
        self.txt_label_MRI_2.setGeometry(1324, 456, 142, 35)
        self.txt_label_MRI_2.setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 8)
        font.setBold(True)  # 设置字体加粗
        self.txt_label_MRI_2.setFont(font)
        self.txt_label_MRI_2.setStyleSheet("QLabel{background:white;}")

        self.start_3 = QPushButton("start", self)
        self.start_3.setFixedSize(50, 35)
        self.start_3.move(506,747)
        self.start_3.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                        "background-color:lightgreen; color: black;font-size: 20px;")

        self.txt_label_3 = QLabel(self)
        self.txt_label_3.setGeometry(556,747, 200, 35)
        self.txt_label_3.setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 8)
        font.setBold(True)  # 设置字体加粗
        self.txt_label_3.setFont(font)
        self.txt_label_3.setStyleSheet("QLabel{background:white;}")

        self.stop_3 = QPushButton("stop", self)
        self.stop_3.setFixedSize(50, 35)
        self.stop_3.move(756, 747)
        self.stop_3.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                  "background-color:lightgreen; color: black;font-size: 20px;")

        self.txt_label_MRI_3 = QLabel(self)
        self.txt_label_MRI_3.setGeometry(806,747, 142, 35)
        self.txt_label_MRI_3.setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 8)
        font.setBold(True)  # 设置字体加粗
        self.txt_label_MRI_3.setFont(font)
        self.txt_label_MRI_3.setStyleSheet("QLabel{background:white;}")

        self.start_4 = QPushButton("start", self)
        self.start_4.setFixedSize(50, 35)
        self.start_4.move(1024,747)
        self.start_4.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                        "background-color:lightgreen; color: black;font-size: 20px;")

        self.txt_label_4 = QLabel(self)
        self.txt_label_4.setGeometry(1074,747, 200, 35)
        self.txt_label_4.setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 8)
        font.setBold(True)  # 设置字体加粗
        self.txt_label_4.setFont(font)
        self.txt_label_4.setStyleSheet("QLabel{background:white;}")

        self.stop_4 = QPushButton("stop", self)
        self.stop_4.setFixedSize(50, 35)
        self.stop_4.move(1274, 747)
        self.stop_4.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                  "background-color:lightgreen; color: black;font-size: 20px;")

        self.txt_label_MRI_4 = QLabel(self)
        self.txt_label_MRI_4.setGeometry(1324,747, 142, 35)
        self.txt_label_MRI_4.setAlignment(Qt.AlignCenter)
        font = QFont("Times New Roman", 8)
        font.setBold(True)  # 设置字体加粗
        self.txt_label_MRI_4.setFont(font)
        self.txt_label_MRI_4.setStyleSheet("QLabel{background:white;}")


        self.button_predict = QPushButton("Predict", self)
        self.button_predict.setFixedSize(180,90)
        self.button_predict.move(520, 45)
        self.button_predict.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                    "background-color:paleturquoise; color: black;font-size: 32px;")

        self.button_correct = QPushButton("Correct", self)
        self.button_correct.setFixedSize(180,90)
        self.button_correct.move(780, 45)
        self.button_correct.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                    "background-color:paleturquoise; color: black;font-size: 32px;")

        self.button_registering = QPushButton("Registering", self)
        self.button_registering.setFixedSize(180,90)
        self.button_registering.move(1040, 45)
        self.button_registering.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                    "background-color:paleturquoise; color: black;font-size: 32px;")

        self.button_save = QPushButton("visualization", self)
        self.button_save.setFixedSize(180,90)
        self.button_save.move(1300, 45)
        self.button_save.setStyleSheet("font-family: Times New Roman; font-weight: bold; text-align: center;"
                                              "background-color:paleturquoise; color: black;font-size: 32px;")


    def sece_LSCM(self):
        def makeDir(path):
            if not os.path.exists(path):
                if not os.path.isfile(path):
                    os.makedirs(path)

        makeDir("picture")
        makeDir("picture/LSCM")
        self.videoName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "open", "./",
                                                                  "All Files (*);; *.mp4;;*avi")
        self.txt_label_1.setText(self.videoName)
    def sece_LSCM_MRI(self):
        self.videoName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "open", "./",
                                                                  "All Files (*);; *.jpg;;*png;;*.bmp")
        self.txt_label_MRI_1.setText(self.videoName)
        pixmap = QPixmap( self.videoName)
        # 设置 QLabel 的大小策略，以确保图片按原比例显示
        self.label_MRI_1.setScaledContents(True)

        # 将 QPixmap 设置到 QLabel
        self.label_MRI_1.setPixmap(pixmap)
    def slotStart_LSCM(self):
        videoName = self.txt_label_1.text()
        self.cap_LSCM = cv2.VideoCapture(videoName)
        self.timer_camera_LSCM.start(100)
        self.timer_camera_LSCM.timeout.connect(self.openFrame_LSCM)
    def slotStop_LSCM(self):
        """ Slot function to stop the programme
            """
        if self.cap_LSCM != []:
            self.cap_LSCM.release()
            self.timer_camera_LSCM.stop()  # 停止计时器
    def openFrame_LSCM(self):
        if (self.cap_LSCM.isOpened()):
            ret, self.frame_LSCM = self.cap_LSCM.read()
            if ret:
                frame = cv2.cvtColor(self.frame_LSCM, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QtGui.QImage(frame.data, width, height, bytesPerLine,
                                 QtGui.QImage.Format_RGB888).scaled(self.label_1.width(), self.label_1.height())
                self.label_1.setPixmap(QtGui.QPixmap.fromImage(q_image))
            else:
                self.cap_LSCM.release()
                self.timer_camera_LSCM.stop()  # 停止计时器
    def start_task_LSCM(self):
        self.button_1.setDisabled(True)
        self.worker_thread = WorkerThread(self.txt_label_1.text(),"picture/LSCM")
        self.worker_thread.progress_signal.connect(self.update_progress_bar_LSCM)
        self.worker_thread.start()  # 启动线程

    def update_progress_bar_LSCM(self, value):
        self.progressBar1.setValue(value)





    def sece_LSSCap(self):
        def makeDir(path):
            if not os.path.exists(path):
                if not os.path.isfile(path):
                    os.makedirs(path)

        makeDir("picture/LSSCap")
        self.videoName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "open", "./",
                                                                  "All Files (*);; *.mp4;;*avi")
        self.txt_label_2.setText(self.videoName)

    def sece_LSSCap_MRI(self):
        self.videoName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "open", "./",
                                                                  "All Files (*);; *.jpg;;*png;;*.bmp")
        self.txt_label_MRI_2.setText(self.videoName)
        pixmap = QPixmap(self.videoName)
        # 设置 QLabel 的大小策略，以确保图片按原比例显示
        self.label_MRI_2.setScaledContents(True)

        # 将 QPixmap 设置到 QLabel
        self.label_MRI_2.setPixmap(pixmap)
    def slotStart_LSSCap(self):
        videoName = self.txt_label_2.text()
        self.cap_LSSCap = cv2.VideoCapture(videoName)
        self.timer_camera_LSSCap.start(100)
        self.timer_camera_LSSCap.timeout.connect(self.openFrame_LSSCap)
    def slotStop_LSSCap(self):
        """ Slot function to stop the programme
            """
        if self.cap_LSSCap != []:
            self.cap_LSSCap.release()
            self.timer_camera_LSSCap.stop()  # 停止计时器
    def openFrame_LSSCap(self):
        if (self.cap_LSSCap.isOpened()):
            ret, self.frame_LSSCap = self.cap_LSSCap.read()
            if ret:
                frame = cv2.cvtColor(self.frame_LSSCap, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QtGui.QImage(frame.data, width, height, bytesPerLine,
                                 QtGui.QImage.Format_RGB888).scaled(self.label_2.width(), self.label_2.height())
                self.label_2.setPixmap(QtGui.QPixmap.fromImage(q_image))
            else:
                self.cap_LSSCap.release()
                self.timer_camera_LSSCap.stop()  # 停止计时器
    def start_task_LSSCap(self):
        self.button_2.setDisabled(True)
        self.worker_thread = WorkerThread(self.txt_label_2.text(),"picture/LSSCap")
        self.worker_thread.progress_signal.connect(self.update_progress_bar_LSSCap)
        self.worker_thread.start()  # 启动线程

    def update_progress_bar_LSSCap(self, value):
        self.progressBar2.setValue(value)



    def sece_RSCM(self):
        def makeDir(path):
            if not os.path.exists(path):
                if not os.path.isfile(path):
                    os.makedirs(path)

        makeDir("picture/RSCM")
        self.videoName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "open", "./",
                                                                  "All Files (*);; *.mp4;;*avi")
        self.txt_label_3.setText(self.videoName)

    def sece_RSCM_MRI(self):
        self.videoName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "open", "./",
                                                                  "All Files (*);; *.jpg;;*png;;*.bmp")
        self.txt_label_MRI_3.setText(self.videoName)
        pixmap = QPixmap( self.videoName)
        # 设置 QLabel 的大小策略，以确保图片按原比例显示
        self.label_MRI_3.setScaledContents(True)

        # 将 QPixmap 设置到 QLabel
        self.label_MRI_3.setPixmap(pixmap)
    def slotStart_RSCM(self):
        videoName = self.txt_label_3.text()
        self.cap_RSCM = cv2.VideoCapture(videoName)
        self.timer_camera_RSCM.start(100)
        self.timer_camera_RSCM.timeout.connect(self.openFrame_RSCM)
    def slotStop_RSCM(self):
        """ Slot function to stop the programme
            """
        if self.cap_RSCM != []:
            self.cap_RSCM.release()
            self.timer_camera_RSCM.stop()  # 停止计时器
    def openFrame_RSCM(self):
        if (self.cap_RSCM.isOpened()):
            ret, self.frame_RSCM = self.cap_RSCM.read()
            if ret:
                frame = cv2.cvtColor(self.frame_RSCM, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QtGui.QImage(frame.data, width, height, bytesPerLine,
                                 QtGui.QImage.Format_RGB888).scaled(self.label_3.width(), self.label_3.height())
                self.label_3.setPixmap(QtGui.QPixmap.fromImage(q_image))
            else:
                self.cap_RSCM.release()
                self.timer_camera_RSCM.stop()  # 停止计时器
    def start_task_RSCM(self):
        self.button_3.setDisabled(True)
        self.worker_thread = WorkerThread(self.txt_label_3.text(),"picture/RSCM")
        self.worker_thread.progress_signal.connect(self.update_progress_bar_RSCM)
        self.worker_thread.start()  # 启动线程

    def update_progress_bar_RSCM(self, value):
        self.progressBar3.setValue(value)




    def sece_RSSCap(self):
        def makeDir(path):
            if not os.path.exists(path):
                if not os.path.isfile(path):
                    os.makedirs(path)

        makeDir("picture/RSSCap")
        self.videoName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "open", "./",
                                                                  "All Files (*);; *.mp4;;*avi")
        self.txt_label_4.setText(self.videoName)

    def sece_RSSCap_MRI(self):
        self.videoName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "open", "./",
                                                                  "All Files (*);; *.jpg;;*png;;*.bmp")
        self.txt_label_MRI_4.setText(self.videoName)
        pixmap = QPixmap( self.videoName)
        # 设置 QLabel 的大小策略，以确保图片按原比例显示
        self.label_MRI_4.setScaledContents(True)

        # 将 QPixmap 设置到 QLabel
        self.label_MRI_4.setPixmap(pixmap)

    def sece_MRI(self):
        self.videoName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "open", "./",
                                                                  "All Files (*);; *.jpg;;*png;;*.bmp")
        self.txt_label.setText(self.videoName)
        pixmap = QPixmap( self.videoName)
        # 设置 QLabel 的大小策略，以确保图片按原比例显示
        self.label_7.setScaledContents(True)

        # 将 QPixmap 设置到 QLabel
        self.label_7.setPixmap(pixmap)
    def slotStart_RSSCap(self):
        videoName = self.txt_label_4.text()
        self.cap_RSSCap = cv2.VideoCapture(videoName)
        self.timer_camera_RSSCap.start(100)
        self.timer_camera_RSSCap.timeout.connect(self.openFrame_RSSCap)
    def slotStop_RSSCap(self):
        """ Slot function to stop the programme
            """
        if self.cap_RSSCap != []:
            self.cap_RSSCap.release()
            self.timer_camera_RSSCap.stop()  # 停止计时器
    def openFrame_RSSCap(self):
        if (self.cap_RSSCap.isOpened()):
            ret, self.frame_RSSCap = self.cap_RSSCap.read()
            if ret:
                frame = cv2.cvtColor(self.frame_RSSCap, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QtGui.QImage(frame.data, width, height, bytesPerLine,
                                 QtGui.QImage.Format_RGB888).scaled(self.label_4.width(), self.label_4.height())
                self.label_4.setPixmap(QtGui.QPixmap.fromImage(q_image))
            else:
                self.cap_RSSCap.release()
                self.timer_camera_RSSCap.stop()  # 停止计时器
    def start_task_RSSCap(self):
        self.button_4.setDisabled(True)
        self.worker_thread = WorkerThread(self.txt_label_4.text(),"picture/RSSCap")
        self.worker_thread.progress_signal.connect(self.update_progress_bar_RSSCap)
        self.worker_thread.start()  # 启动线程

    def update_progress_bar_RSSCap(self, value):
        self.progressBar4.setValue(value)


    def start_task_seg_LSCM(self):
        self.button_predict.setDisabled(True)
        self.worker_thread_seg_LSCM = WorkerThread_seg("LSCM")
        self.worker_thread_seg_LSCM.progress_signal.connect(self.update_progress_bar_seg_LSCM)
        self.worker_thread_seg_LSCM.finished.connect(self.start_task_seg_LSSCap_after_LSCM)
        self.txt_label_1.setText("img/LSCM.mp4")
        self.worker_thread_seg_LSCM.start()  # 启动线程


    def update_progress_bar_seg_LSCM(self, value):
        self.progressBar1.setValue(value)
    def update_progress_bar_seg_LSSCap(self, value):
        self.progressBar2.setValue(value)
    def update_progress_bar_seg_RSCM(self, value):
        self.progressBar3.setValue(value)
    def update_progress_bar_seg_RSSCap(self, value):
        self.progressBar4.setValue(value)
    def start_task_seg_LSSCap_after_LSCM(self):
        # 第一个任务已完成，现在启动第二个任务
        self.worker_thread_seg_LSSCap = WorkerThread_seg("LSSCap")
        self.worker_thread_seg_LSSCap.progress_signal.connect(self.update_progress_bar_seg_LSSCap)
        self.worker_thread_seg_LSSCap.finished.connect(self.start_task_seg_RSCM_after_LSSCap)
        self.txt_label_2.setText("img/LSSCap.mp4")
        self.worker_thread_seg_LSSCap.start()  # 启动线程
    def start_task_seg_RSCM_after_LSSCap(self):
        # 第一个任务已完成，现在启动第二个任务
        self.worker_thread_seg_RSCM = WorkerThread_seg("RSCM")
        self.worker_thread_seg_RSCM.progress_signal.connect(self.update_progress_bar_seg_RSCM)
        self.worker_thread_seg_RSCM.finished.connect(self.start_task_seg_RSSCap_after_RSCM)
        self.txt_label_3.setText("img/RSCM.mp4")
        self.worker_thread_seg_RSCM.start()  # 启动线程

    def start_task_seg_RSSCap_after_RSCM(self):
        # 第一个任务已完成，现在启动第二个任务
        self.worker_thread_seg_RSSCap = WorkerThread_seg("RSSCap")
        self.worker_thread_seg_RSSCap.progress_signal.connect(self.update_progress_bar_seg_RSSCap)
        self.worker_thread_seg_RSSCap.finished.connect(self.on_both_tasks_finished)
        self.txt_label_4.setText("img/RSSCap.mp4")
        self.worker_thread_seg_RSSCap.start()

    def on_both_tasks_finished(self):
        # 两个任务都已完成，重新启用按钮
        self.button_predict.setDisabled(False)






    def start_task_cor_1(self):
        self.button_correct.setDisabled(True)
        self.worker_thread_cor = WorkerThread_cor("LSCM")
        self.worker_thread_cor.progress_signal.connect(self.update_progress_bar_cor_1)
        self.worker_thread_cor.finished.connect(self.start_second_LSSCap_after_LSCM)
        self.txt_label_1.setText("correct/LSCM.mp4")
        self.worker_thread_cor.start()  # 启动线程

    def start_second_LSSCap_after_LSCM(self):
        # 第一个任务已完成，现在创建并启动第二个线程
        self.worker_thread_cor = WorkerThread_cor("LSSCap")
        self.worker_thread_cor.progress_signal.connect(self.update_progress_bar_cor_2)
        self.worker_thread_cor.finished.connect(self.start_second_RSCM_after_LSSCap)
        self.txt_label_2.setText("correct/LSSCap.mp4")
        self.worker_thread_cor.start()

    def start_second_RSCM_after_LSSCap(self):
        # 第一个任务已完成，现在创建并启动第二个线程
        self.worker_thread_cor = WorkerThread_cor("RSCM")
        self.worker_thread_cor.progress_signal.connect(self.update_progress_bar_cor_3)
        self.worker_thread_cor.finished.connect(self.start_second_RSSCap_after_RSCM)
        #self.worker_thread_cor_2.finished.connect(self.on_both_tasks_finished)
        self.txt_label_3.setText("correct/RSCM.mp4")
        self.worker_thread_cor.start()

    def start_second_RSSCap_after_RSCM(self):
        # 第一个任务已完成，现在创建并启动第二个线程
        self.worker_thread_cor = WorkerThread_cor("RSSCap")
        self.worker_thread_cor.progress_signal.connect(self.update_progress_bar_cor_4)
        #self.worker_thread_cor.finished.connect(self.on_both_tasks_finished)
        self.txt_label_4.setText("correct/RSSCap.mp4")
        self.worker_thread_cor.start()
    def update_progress_bar_cor_1(self, value):
        self.progressBar1.setValue(value)

    def update_progress_bar_cor_2(self, value):
        self.progressBar2.setValue(value)

    def update_progress_bar_cor_3(self, value):
        self.progressBar3.setValue(value)

    def update_progress_bar_cor_4(self, value):
        self.progressBar4.setValue(value)

    def on_both_tasks_finished(self):
        # 两个任务都已完成，重新启用按钮
        self.button_correct.setDisabled(False)


    def start_task_reg_1(self):
        self.button_correct.setDisabled(True)
        image_path1 = self.txt_label_MRI_1.text()
        image_path2 = self.txt_label.text()
        self.worker_thread_reg = WorkerThread_reg("LSCM",image_path1,image_path2)
        self.worker_thread_reg.progress_signal.connect(self.update_progress_bar_reg_LSCM)
        self.worker_thread_reg.finished.connect(self.start_second_LSSCap_after_LSCM_1)
        self.txt_label_1.setText("registering/LSCM.mp4")
        self.worker_thread_reg.start()  # 启动线程

    def update_progress_bar_reg_LSCM(self, value):
        self.progressBar1.setValue(value)

    def start_second_LSSCap_after_LSCM_1(self):
        self.button_correct.setDisabled(True)
        image_path1 = self.txt_label_MRI_2.text()
        image_path2 = self.txt_label.text()
        self.worker_thread_reg = WorkerThread_reg("LSSCap",image_path1,image_path2)
        self.worker_thread_reg.progress_signal.connect(self.update_progress_bar_reg_LSSCap)
        self.worker_thread_reg.finished.connect(self.start_second_RSCM_after_LSSCap_1)
        self.txt_label_2.setText("registering/LSSCap.mp4")
        self.worker_thread_reg.start()  # 启动线程

    def update_progress_bar_reg_LSSCap(self, value):
        self.progressBar2.setValue(value)

    def start_second_RSCM_after_LSSCap_1(self):
        self.button_correct.setDisabled(True)
        image_path1 = self.txt_label_MRI_3.text()
        image_path2 = self.txt_label.text()
        self.worker_thread_reg = WorkerThread_reg("RSCM",image_path1,image_path2)
        self.worker_thread_reg.progress_signal.connect(self.update_progress_bar_reg_RSCM)
        self.worker_thread_reg.finished.connect(self.start_second_RSSCap_after_RSCM_1)
        self.txt_label_3.setText("registering/RSCM.mp4")
        self.worker_thread_reg.start()  # 启动线程

    def update_progress_bar_reg_RSCM(self, value):
        self.progressBar3.setValue(value)

    def start_second_RSSCap_after_RSCM_1(self):
        self.button_correct.setDisabled(True)
        image_path1 = self.txt_label_MRI_4.text()
        image_path2 = self.txt_label.text()
        self.worker_thread_reg = WorkerThread_reg("RSSCap",image_path1,image_path2)
        self.worker_thread_reg.progress_signal.connect(self.update_progress_bar_reg_RSSCap)
        self.txt_label_4.setText("registering/RSSCap.mp4")
        self.worker_thread_reg.start()  # 启动线程

    def update_progress_bar_reg_RSSCap(self, value):
        self.progressBar4.setValue(value)

    def visi(self):
        def makeDir(path):
            if not os.path.exists(path):
                if not os.path.isfile(path):
                    os.makedirs(path)
        makeDir("three_dimensional")
        def rgb_to_label(image):
            h, w, c = image.shape
            label = np.zeros((h, w), dtype=np.uint8)
            label[np.where((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0))] = 0  # 黑色对应类别0
            label[np.where((image[:, :, 0] == 0) & (image[:, :, 1] == 255) & (image[:, :, 2] == 0))] = 1  # 绿色 对应类别2
            label[np.where((image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 255))] = 2  # 紫色对应类别2
            label[np.where((image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0))] = 3  # 红色对应类别3
            label[np.where((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 255))] = 5  # 蓝色对应类别4

            return label

        # 将png图像转为数组形式并填充在三维数组中
        def rgb_numpy(input_path, save_path):
            image_spacing = 1.0
            dirs = os.listdir(input_path)
            img_a = []
            for i in range(0,len(dirs)):
                img_path = os.path.join(input_path, str(i) + ".png")
                print(img_path)
                img = cv2.imread(img_path)
                img = rgb_to_label(img)
                img_array = np.array(img)
                img_a.append(img_array)

            image_shape = img_a[0].shape
            num_images = len(img_a)

            # 根据image_spacing的类型确定三维空间的形状
            if isinstance(image_spacing, (float, int)):
                spacing = (image_spacing,) * 3
            elif len(image_spacing) == 3:
                spacing = tuple(image_spacing)
            else:
                raise ValueError("image_spacing 的长度必须是1（表示等间隔）或3（表示x, y, z方向上的间隔）。")

            # 计算三维空间的形状
            depth, height, width = num_images, image_shape[0], image_shape[1]

            # 初始化三维空间的数据数组
            d_data = np.zeros((depth, height, width), dtype=np.float32)

            # 将二维图像映射到三维空间
            for i, image in enumerate(img_a):
                z = i * spacing[0]
                d_data[i, :, :] = image

            affine = np.eye(4)  # 仿射矩阵，通常用于描述图像的空间变换，这里使用单位矩阵表示没有变换
            nii_image = Nifti1Image(d_data, affine)

            # 保存Nifti1Image对象为NIfTI文件

            nib.save(nii_image, save_path)

        def png_nii_thick(nii_path, output_nii, thickness):
            nii_image = nib.load(nii_path)
            nii_data = nii_image.get_fdata()
            affine = nii_image.affine
            # 获取原始图像的尺寸
            original_shape = nii_data.shape
            original_thickness = original_shape[0]
            # 计算插值因子
            interpolation_factor = thickness / original_thickness
            # 调整数据大小以匹配新的厚度
            # 使用scipy的插值函数来改变数据的大小
            # 注意：这里假设我们要在第一个维度（通常是z轴）上进行插值
            resized_data = zoom(nii_data, (interpolation_factor, 1, 1), order=1)
            # 创建新的NIfTI图像对象
            new_nii_img = nib.Nifti1Image(resized_data, affine)
            # 将新的NIfTI图像保存到文件
            nib.save(new_nii_img, output_nii)

        def smooth_nii(nii_path, save_path):
            # 读取NIfTI文件
            nii_image = nib.load(nii_path)
            image_data = nii_image.get_fdata()
            affine = nii_image.affine
            header = nii_image.header
            sigma = 1.5
            # 应用高斯平滑
            smoothed_data = gaussian_filter(image_data, sigma=sigma)

            # 创建新的NIfTI图像对象
            smoothed_nii_image = nib.Nifti1Image(smoothed_data, affine, header)

            # 保存平滑后的NIfTI文件
            nib.save(smoothed_nii_image, save_path)

        def keshihua(img_path,save_path):
            # 加载.nii.gz文件
            img = nib.load(img_path)
            data = img.get_fdata()

            # 创建一个空的Mayavi场景
            scene = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 600))

            # 显示3D图像数据
            # 注意：这里我们简单地显示了整个数据体，你可能需要根据你的需求来切片或调整显示的数据部分
            src = mlab.pipeline.volume(mlab.pipeline.scalar_field(data))
           # src.volume_property.scalar_opacity_unit_distance = 0.1
            mlab.savefig(save_path,figure=src)
            # 交互模式
            #mlab.show()

        def nii_add(img1, img2):
            # # 检查两个图像的形状是否一致
            # if img1.shape != img2.shape:
            #     print("Images have different shapes and cannot be added directly.")
            #     # 这里可以添加代码进行图像配准，或者处理形状不一致的情况
            #     # ...
            # else:
            #     # 提取数据
                data1 = img1.get_fdata()
                data2 = img2.get_fdata()

                # 逐元素相加
                result_data = data1 + data2

                # 创建新的Nifti1Image对象
                new_img = nib.Nifti1Image(result_data, img1.affine, img1.header)

                # 保存新的nii.gz文件
                # nib.save(new_img, 'result.nii.gz')
                return new_img

        file_path = "registering/LSCM"
        save_path = "three_dimensional"
        save_path2 = save_path +"/" + "LSCM.nii.gz"
        rgb_numpy(file_path, save_path2)
        thickness = 60
        png_nii_thick(save_path2, save_path2, thickness)
        smooth_nii(save_path2,save_path2)
        keshihua(save_path2,save_path +"/" + "LSCM.png")
        self.txt_label_MRI_1.setText(save_path +"/" + "LSCM.png")
        pixmap = QPixmap( save_path +"/" + "LSCM.png")
        # 设置 QLabel 的大小策略，以确保图片按原比例显示
        self.label_MRI_1.setScaledContents(True)

        # 将 QPixmap 设置到 QLabel
        self.label_MRI_1.setPixmap(pixmap)

        file_path = "registering/LSSCap"
        save_path = "three_dimensional"
        save_path2 = save_path +"/" + "LSSCap.nii.gz"
        rgb_numpy(file_path, save_path2)
        thickness = 60
        png_nii_thick(save_path2, save_path2, thickness)
        smooth_nii(save_path2,save_path2)
        keshihua(save_path2,save_path +"/" + "LSSCap.png")
        self.txt_label_MRI_2.setText(save_path +"/" + "LSSCap.png")
        pixmap = QPixmap( save_path +"/" + "LSSCap.png")
        # 设置 QLabel 的大小策略，以确保图片按原比例显示
        self.label_MRI_2.setScaledContents(True)

        # 将 QPixmap 设置到 QLabel
        self.label_MRI_2.setPixmap(pixmap)

        file_path = "registering/RSCM"
        save_path = "three_dimensional"
        save_path2 = save_path + "/" + "RSCM.nii.gz"
        rgb_numpy(file_path, save_path2)
        thickness = 60
        png_nii_thick(save_path2, save_path2, thickness)
        smooth_nii(save_path2, save_path2)
        keshihua(save_path2, save_path + "/" + "RSCM.png")
        self.txt_label_MRI_3.setText(save_path + "/" + "RSCM.png")
        pixmap = QPixmap( save_path +"/" + "RSCM.png")
        # 设置 QLabel 的大小策略，以确保图片按原比例显示
        self.label_MRI_3.setScaledContents(True)

        # 将 QPixmap 设置到 QLabel
        self.label_MRI_3.setPixmap(pixmap)

        file_path = "registering/RSSCap"
        save_path = "three_dimensional"
        save_path2 = save_path + "/" + "RSSCap.nii.gz"
        rgb_numpy(file_path, save_path2)
        thickness = 60
        png_nii_thick(save_path2, save_path2, thickness)
        smooth_nii(save_path2, save_path2)
        keshihua(save_path2, save_path + "/" + "RSSCap.png")
        self.txt_label_MRI_4.setText(save_path + "/" + "RSSCap.png")
        pixmap = QPixmap( save_path +"/" + "RSSCap.png")
        # 设置 QLabel 的大小策略，以确保图片按原比例显示
        self.label_MRI_4.setScaledContents(True)

        # 将 QPixmap 设置到 QLabel
        self.label_MRI_4.setPixmap(pixmap)


        lscm = nib.load(save_path + "/" + "LSCM.nii.gz")
        lsscap = nib.load(save_path + "/" + "LSSCap.nii.gz")
        rscm = nib.load(save_path + "/" + "RSCM.nii.gz")
        rsscap = nib.load(save_path + "/" + "RSSCap.nii.gz")

        result1 = nii_add(lscm, lsscap)
        result2 = nii_add(rscm, rsscap)
        result = nii_add(result1, result2)
        save1 = save_path + "/" + "result.nii.gz"
        save = save_path + "/" + "result.png"
        nib.save(result, save1)
        keshihua(save1, save)
        self.txt_label_5.setText(save)
        pixmap = QPixmap(save_path + "/" + "result.png")
        # 设置 QLabel 的大小策略，以确保图片按原比例显示
        self.label_6.setScaledContents(True)
        # 将 QPixmap 设置到 QLabel
        self.label_6.setPixmap(pixmap)


    def save(self):
        self.image, _ = QtWidgets.QFileDialog.getOpenFileName(None, "open", "./",
                                                                " *.png;;*jpg;;*.bmp")
        self.txt_label_4.setText(self.image)
        save_image = self.txt_label_5.text()
        old_1 = "three_dimensional" +"/" + "result.png"
        new_1 = save_image
        shutil.move(old_1, new_1)





    # 界面关闭事件，询问用户是否关闭
    def closeEvent(self, event):
        def delete_if_exists(path):
            """
            删除给定的路径（文件或文件夹）。
            如果路径不存在，则忽略。
            """
            if os.path.exists(path):
                if os.path.isfile(path) or os.path.islink(path):
                    # 如果是文件或链接，直接删除
                    os.unlink(path)
                elif os.path.isdir(path):
                    # 如果是文件夹，递归删除
                    shutil.rmtree(path)
                else:
                    print(f"Error: {path} is not a recognized file or directory")
                    return
            else:
                # 路径不存在，忽略
                print(f"Warning: {path} does not exist, skipping.")
        reply = QMessageBox.question(self, 'quit', "Do you want to exit this interface?？",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            delete_if_exists("img")
            delete_if_exists("correct")
            delete_if_exists("test.png")
            delete_if_exists("picture")
            delete_if_exists("registering")
            delete_if_exists("test.png")
            self.close()
            event.accept()
        else:
            event.ignore()




if __name__ == '__main__':
    # 创建应用程序实例
    app = QApplication(sys.argv)
    # 创建界面实例
    window = MainWindow()
    # 设置原始图片和处理后的图片
    # 显示界面
    window.show()
    # 运行应用程序
    sys.exit(app.exec_())
