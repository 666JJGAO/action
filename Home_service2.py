import numpy as np
from Pyrealsense import pyrealsense
from KeyPointDetection import Keypoint
import cv2
import time
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import serial
import threading
import datetime
from collections import Counter

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from test03.srv import pose, poseRequest, poseResponse
from sensor_msgs.msg import LaserScan

import dlib
import pandas as pd
import os
import logging
from PIL import Image, ImageDraw, ImageFont
from action_recogion import FallDetection, my_pose_point, cv2_add_chinese_text
from collections import deque
from collections import defaultdict
from playsound import playsound

def play_wav_file(file_name):
    playsound(file_name)

# Dlib 正向人脸检测器 / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib 人脸 landmark 特征点检测器 / Get face landmarks
predictor = dlib.shape_predictor('/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet 人脸识别模型，提取 128D 的特征矢量 / Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

key_MAX_FRAMES = 30
key_list = deque(maxlen=key_MAX_FRAMES)
# 定义一个事件对象
play_finished = threading.Event()

def play_audio(audio_file):
    play_wav_file(audio_file)
    play_finished.set()

def check_playback_status():
    if play_finished.is_set():
        return 1  # 播放完毕
    else:
        return 0  # 播放未完

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(seg1, seg2):
    A, B = seg1
    C, D = seg2
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


class BehavioralDetection:
    def __init__(self, Body_detection_path, serial_port):
        self.pyrs = pyrealsense()
        self.BodyDetector = Keypoint(Body_detection_path, show=False, draw=True, fps=False, need_bone=True)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.goal = MoveBaseGoal()
        self.ser = serial.Serial(serial_port, 115200, timeout=2)
        self.pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.msg = Twist()
        self.rate = rospy.Rate(10)
        self.PI = np.pi
        self.start_time = time.time()
        self.arrive_target_distance = 0
        self.h_mechanical_arm = False

        self.face_feature_known_list = []  # 用来存放所有录入人脸特征的数组 / Save the features of faces in database
        self.face_name_known_list = []  # 存储录入人脸名字 / Save the name of faces in database

        self.current_frame_face_cnt = 0  # 存储当前摄像头中捕获到的人脸数 / Counter for faces in current frame
        self.current_frame_face_feature_list = []  # 存储当前摄像头中捕获到的人脸特征 / Features of faces in current frame
        self.current_frame_face_name_list = []  # 存储当前摄像头中捕获到的所有人脸的名字 / Names of faces in current frame
        self.current_frame_face_name_position_list = []  # 存储当前摄像头中捕获到的所有人脸的名字坐标 / Positions of faces in current frame

        # Update FPS
        self.fps = 0  # FPS of current frame
        self.fps_show = 0  # FPS per second
        self.frame_start_time = 0
        self.frame_cnt = 0

        self.font = cv2.FONT_ITALIC

        track_history = defaultdict(lambda: [])
        self.left_track = track_history[0]
        self.right_track = track_history[1]
        '''
        初始化装前后30帧的关键点
        '''
        # 最大帧数
        ACTION_MODEL_MAX_FRAMES = 30
        # deque（双端队列）是Python标准库collections模块中的一种数据结构。它类似于列表（list），但具有在两端有效添加和删除元素的特性。
        self.joints_list = deque(maxlen=ACTION_MODEL_MAX_FRAMES)

        '''
        初始化 模型和字体
        '''
        self.actin = FallDetection('./weights/tsstg-model.pth')
        self.word_path = './fonts/MSYH.ttc'
        self.model = YOLO("yolov8x-pose.pt")
        self.My_pose_point = my_pose_point()

        self.action_re_switch = False
        self.hand_switch = True
        self.next_action = False
        self.wait_switch = True

        self.success_time_begin_0_switch = True  # 举手
        self.success_time_begin_1_switch = True  # 挥手
        self.success_time_begin_2_switch = True  # 会双手
        self.success_time_begin_3_switch = True  # 打电话
        self.success_time_begin_4_switch = True  # 叉手
        self.false_time_begin_switch = True
        self.next = True

    def update_fps(self):
        now = time.time()
        # 每秒刷新 fps / Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

        # 生成的 cv2 window 上面添加说明文字 / PutText on cv2 window

    def draw_note(self, img_rd):
        cv2.putText(img_rd, "Face Recognizer", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps_show.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0),
                    1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def get_face_database(self):
        if os.path.exists(
                "/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/data/features_all.csv"):
            path_features_known_csv = "/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_feature_known_list.append(features_someone_arr)
            logging.info("Faces in Database：%d", len(self.face_feature_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

        # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features

    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    def draw_name(self, img_rd):
        # 在人脸框下面写人脸名字 / Write names under rectangle
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for i in range(self.current_frame_face_cnt):
            # cv2.putText(img_rd, self.current_frame_face_name_list[i], self.current_frame_face_name_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            draw.text(xy=self.current_frame_face_name_position_list[i], text=self.current_frame_face_name_list[i],
                      fill=(255, 255, 0))
            img_rd = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_rd


    def quaternion2euler(self, quaternion):
        r = R.from_quat(quaternion)
        euler = r.as_euler('xyz', degrees=True)
        if euler[2] < 0:
            euler[2] += 360
        return euler[2]

    def euler2quaternion(self, euler_z):
        r = R.from_euler('z', euler_z, degrees=True)
        quaternion = r.as_quat()
        if quaternion[3] < 0:
            quaternion = [-quaternion[2], -quaternion[3]]
        else:
            quaternion = [quaternion[2], quaternion[3]]
        return quaternion

    def go_into_or_out_room(self, point):
        self.go_to_another_point(point)
        for _ in range(20):
            self.msg.linear.x = 0.2
            self.msg.angular.z = 0
            self.pub.publish(self.msg)
            self.rate.sleep()
            time.sleep(0.3)


    def initalization_speed(self):
        self.msg.linear.x = 0.02
        n = 0
        while 10 - n:
            for i in range(4):
                self.pub.publish(self.msg)
                time.sleep(0.03)
            self.msg.linear.x += 0.01
            n += 1

    # 用于获取机器人的初始位姿
    def getIniPose(self):
        IniPose = rospy.wait_for_message("/amcl_pose", PoseWithCovarianceStamped, timeout=None)
        x = IniPose.pose.pose.position.x
        y = IniPose.pose.pose.position.y
        z = IniPose.pose.pose.orientation.z
        w = IniPose.pose.pose.orientation.w
        # 将四元数转换为欧拉角
        quaternion = [0.0, 0.0, z, w]
        euler = self.quaternion2euler(quaternion)
        print("机器人现在的位置为{}".format((x, y, euler)))
        return x, y, z, w, euler

    def shoulder_head(self, shoulder, not_empty_head, original_head, epsilon=1e-10):
        _, coordinate = self.pyrs.get_3d_camera_coordinate(shoulder)
        if not coordinate:
            print("检测单一肩膀时没有检测到深度值")
            return None
        shoulderx, shouldery, shoulderz = round(coordinate[0], 3), round(coordinate[1], 3), round(coordinate[2], 3)
        if not_empty_head:
            coordinate, original_head = not_empty_head[0], original_head[0]
            headx, heady, headz = round(coordinate[0], 3), round(coordinate[1], 3), round(coordinate[2], 3)

            if headx == 0 and headz == 0:
                # print("没有检测到头部的深度信息 试图将头的点向肩膀处靠近")
                distance = 1
                # 计算方向向量
                direction_vector = (shoulder[0] - original_head[0], shoulder[1] - original_head[1])
                # 计算单位向量
                magnitude = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
                unit_vector = (direction_vector[0] / magnitude, direction_vector[1] / magnitude)
                while 11 - distance:
                    # 计算新的点坐标
                    original_head = (
                        int(original_head[0] + distance * unit_vector[0]),
                        int(original_head[1] + distance * unit_vector[1]))
                    _, coordinate = self.pyrs.get_3d_camera_coordinate(original_head)
                    headx, heady, headz = round(coordinate[0], 3), round(coordinate[1], 3), round(coordinate[2], 3)
                    # print(f'headx, heady, headz:{headx, heady, headz}')
                    if headx == 0 and headz == 0:
                        distance += 1
                    elif headx != 0 and headz != 0:
                        # print(f"经过{11 - distance}次靠近后 已经可以获取到头部深度信息")
                        break
                if distance == 11:
                    return None
            # 根据头与单肩距离中心的大小来确定旋转的角度取正还是取负
            if headx < shoulderx:
                # theta = np.arccos(0.15 / hypotenuse)
                # theta = 50.194429
                theta = 0.8760580522085248
                # print(f"头在左")
            else:
                # theta = -np.arccos(0.15 / hypotenuse)
                # theta = -50.194429
                theta = -0.8760580522085248
                # print(f"头在右边！！！")
            a, b = np.array([headx, headz]), np.array([shoulderx, shoulderz])
            # hypotenuse = np.linalg.norm(a - b)
            # 创建旋转矩阵
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            # 旋转向量
            v_rotated = np.dot(rotation_matrix, a - b)
            v_rotated[np.abs(v_rotated) < epsilon] = 0
            # 单位化
            # v_rotated /= np.linalg.norm(v_rotated)
            orientation = np.arctan2(v_rotated[1], v_rotated[0])
            # print(
            #     f'a-b:{a - b},head:{headx, headz}, shoulder:{shoulderx, shoulderz}\norientation:{np.degrees(orientation)}, v_rotated:{v_rotated}')

            # 弧度转角度
            if theta > 0:
                # angle_degrees = np.degrees(orientation - np.pi)
                angle_degrees = orientation - np.pi
            else:
                # angle_degrees = np.degrees(orientation + np.pi)
                angle_degrees = orientation + np.pi
            # angle_degrees = np.degrees(np.pi - orientation)
            return angle_degrees
        else:
            return None

    def determine_personnel_orientation(self, nose, eye1, eye2, shoulder1, shoulder2, frame, max_shoulder_dis=0.41,
                                        min_shoulder_dis=0.3):
        # 格式[,] [,] [,] [,] [,]
        not_empty_head, original_head = [], []
        # 检查有没有获取到头部的深度信息
        for lst in [nose, eye1, eye2]:
            if lst:
                original_head.append(lst)
                _, coordinate = self.pyrs.get_3d_camera_coordinate(nose)
                not_empty_head.append(coordinate)
        # not_empty_head = [lst for lst in [d_nose, d_eye1, d_eye2] if lst]
        # original_head = [lst for lst in [nose, eye1, eye2] if lst]
        # 如果检测到了双肩
        if shoulder1[0] and shoulder2[0]:
            # turn = True if not not_empty_head else False
            dis1, coordinate1 = self.pyrs.get_3d_camera_coordinate(shoulder1)
            dis2, coordinate2 = self.pyrs.get_3d_camera_coordinate(shoulder2)
            if not coordinate1 or not coordinate2:
                # print("yolo检测到了双肩，但是深度相机没有检测到双肩中的某一个深度值")
                # 接下来的操作是将yolo检测到的双肩的点相互靠近 看能不能获取到肩附近的深度信息
                distance = 20
                # 计算中点
                # mid_point = ((shoulder1[0] + shoulder2[0]) // 2, (shoulder1[1] + shoulder2[1]) // 2)
                # 计算方向向量
                direction_vector = (shoulder2[0] - shoulder1[0], shoulder2[1] - shoulder1[1])
                # 计算单位向量
                magnitude = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
                unit_vector = (direction_vector[0] / magnitude, direction_vector[1] / magnitude)
                while distance:
                    # 计算新的点坐标
                    shoulder1_copy = (
                        int(shoulder1[0] + distance * unit_vector[0]), int(shoulder1[1] + distance * unit_vector[1]))
                    shoulder2_copy = (
                        int(shoulder2[0] - distance * unit_vector[0]), int(shoulder2[1] - distance * unit_vector[1]))
                    dis1, coordinate1 = self.pyrs.get_3d_camera_coordinate(shoulder1_copy)
                    dis2, coordinate2 = self.pyrs.get_3d_camera_coordinate(shoulder2_copy)
                    # 如果任然没有检测到深度信息 再次缩小双肩的检测距离
                    if not coordinate1 or not coordinate2:
                        distance -= 2
                    elif coordinate1 and coordinate2:
                        break
                # 如果缩小了10次双肩的距离仍没有检测到深度信息 返回角度-999(也就是人垂直于相机 脸朝左)
                if not distance:
                    return 0
            shoulder1x, shoulder1y, shoulder1z = round(coordinate1[0], 3), round(coordinate1[1], 3), round(
                coordinate1[2],
                3)
            shoulder2x, shoulder2y, shoulder2z = round(coordinate2[0], 3), round(coordinate2[1], 3), round(
                coordinate2[2],
                3)
            # 计算双肩的距离
            shoulder_distance = (((shoulder1x - shoulder2x) ** 2) + ((shoulder1z - shoulder2z) ** 2)) ** 0.5
            # 如果双肩的距离在合理范围之内
            if min_shoulder_dis <= shoulder_distance <= max_shoulder_dis:
                # print(f'双肩的距离:{round(shoulder_distance, 3)}m 是合理值')
                x1, x2, y1, y2 = shoulder1x, shoulder2x, shoulder1z, shoulder2z
                # point0 = np.array([(x2 + x1) / 2, (y2 + y1) / 2])  # 双肩的中点
                normal_vector = -np.array([y1 - y2, x2 - x1])  # 双肩的法向量
                # 向量单位化
                # normal_vector /= (((normal_vector[0] ** 2) + (normal_vector[1] ** 2)) ** 0.5)
                # 判断单位向量的方向 先选取朝向向相机的那个
                # normal_vector = normal_vector if np.linalg.norm(point0 + normal_vector) < np.linalg.norm(
                #     point0 - normal_vector) else -normal_vector
                # # 如果有头部信息 朝向不变 没有头部信息则改变向量方向
                # if turn:
                #     normal_vector = -normal_vector
                angle_degrees = np.arctan2(normal_vector[1], normal_vector[0])
                # angle_degrees = np.degrees(orientation)
            elif shoulder_distance > max_shoulder_dis:
                # print(f"双肩距离过大:{round(shoulder_distance, 3)}m 正在缩减双肩距离")
                # 缩小双肩的距离10次 一共是40个像素点
                distance = 20
                # 计算中点
                # mid_point = ((shoulder1[0] - shoulder2[0]) // 2, (shoulder1[1] - shoulder2[1]) // 2)
                # 计算方向向量
                direction_vector = (shoulder2[0] + shoulder1[0], shoulder2[1] + shoulder1[1])
                # 计算单位向量
                magnitude = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
                unit_vector = (direction_vector[0] / magnitude, direction_vector[1] / magnitude)
                while distance:
                    # 计算新的点坐标
                    shoulder1_copy = (
                        int(shoulder1[0] + distance * unit_vector[0]), int(shoulder1[1] + distance * unit_vector[1]))
                    shoulder2_copy = (
                        int(shoulder2[0] - distance * unit_vector[0]), int(shoulder2[1] - distance * unit_vector[1]))
                    dis1, coordinate1 = self.pyrs.get_3d_camera_coordinate(shoulder1_copy)
                    dis2, coordinate2 = self.pyrs.get_3d_camera_coordinate(shoulder2_copy)
                    if not coordinate1 or not coordinate2:
                        distance -= 2
                    elif coordinate1 and coordinate2:
                        shoulder1x, shoulder1y, shoulder1z = round(coordinate1[0], 3), round(coordinate1[1], 3), round(
                            coordinate1[2], 3)
                        shoulder2x, shoulder2y, shoulder2z = round(coordinate2[0], 3), round(coordinate2[1], 3), round(
                            coordinate2[2], 3)
                        # 计算双肩的距离
                        shoulder_distance = (((shoulder1x - shoulder2x) ** 2) + ((shoulder1z - shoulder2z) ** 2)) ** 0.5
                        if min_shoulder_dis <= shoulder_distance <= max_shoulder_dis:
                            # print(
                            #     f'双肩的距离经过{(22 - distance) // 2}次缩小 已经变得合理了:{round(shoulder_distance, 3)}m')
                            x1, x2, y1, y2 = shoulder1x, shoulder2x, shoulder1z, shoulder2z
                            # point0 = np.array([(x2 + x1) / 2, (y2 + y1) / 2])  # 双肩的中点
                            normal_vector = -np.array([y1 - y2, x2 - x1])  # 双肩的法向量
                            # 向量单位化
                            # normal_vector /= (((normal_vector[0] ** 2) + (normal_vector[1] ** 2)) ** 0.5)
                            # 判断单位向量的方向 先选取朝向向相机的那个
                            # normal_vector = normal_vector if np.linalg.norm(point0 + normal_vector) < np.linalg.norm(
                            #     point0 + normal_vector) else -normal_vector
                            # # 如果有头部信息 朝向不变 没有头部信息则改变向量方向
                            # if turn:
                            #     normal_vector = -normal_vector
                            angle_degrees = np.arctan2(normal_vector[1], normal_vector[0])
                            # angle_degrees = np.degrees(orientation)
                            break
                        elif shoulder_distance < min_shoulder_dis:
                            distance = 0
                        else:
                            distance -= 2
                    # print((20 - distance) // 2, f'{round(shoulder_distance, 3)}m')
                if not distance:
                    # print(f'双肩距离经过10次缩减后任然不合理:{round(shoulder_distance, 3)}m 正在采用另一种方法测量')
                    # 挑选离相机最近的那个肩部点
                    shoulder = shoulder1 if shoulder1z < shoulder2z else shoulder2
                    # if shoulder1z < shoulder2z:
                    #     shoulder, left = shoulder1, True
                    # else:
                    #     shoulder, left = shoulder2, False
                    # print(
                    #     f'shoulder1xyz:{shoulder1x, shoulder1y, shoulder1z}, shoulder2xyz:{shoulder2x, shoulder2y, shoulder2z}')
                    # 如果双肩距离不正常是因为相机没有获取到深度信息 (没有深度信息的肩部点为(0.0, 0.0, 0.0))
                    # 那就根据(左、右)肩与头的关系来判断角度
                    if (not shoulder2x and not shoulder2y and not shoulder2z) or (
                            not shoulder1x and not shoulder1y and not shoulder1z):
                        # 左肩为0且有头部信息
                        if shoulder1x == 0 and not_empty_head:
                            # 头比右肩近
                            if not_empty_head[0][2] < shoulder2z:
                                return 0
                            # 右肩近
                            elif not_empty_head[0][2] > shoulder2z:
                                return 180
                        # 右肩为0且有头部信息
                        elif shoulder2x == 0 and not_empty_head:
                            # 左肩近
                            if not_empty_head[0][2] > shoulder2x:
                                return 0
                            # 头近
                            elif not_empty_head[0][2] < shoulder2x:
                                return 180
                        # 如果两肩均没有深度信息 返回None
                        return None
                    cv2.circle(frame, (int(shoulder[0]), int(shoulder[1])), 10, (255, 100, 20), -1)
                    # 肩都有深度信息 但肩距不对 则用头部信息和离相机更近的肩判断方位
                    angle_degrees = self.shoulder_head(shoulder, not_empty_head, original_head)
                    return angle_degrees
            elif shoulder_distance < min_shoulder_dis:
                # print(f"双肩距离小于正常值:{round(shoulder_distance, 3)}m 正在扩大双肩的距离")
                distance = 1
                # 计算中点
                # mid_point = ((shoulder1[0] + shoulder2[0]) // 2, (shoulder1[1] + shoulder2[1]) // 2)
                # 计算方向向量
                direction_vector = (shoulder2[0] - shoulder1[0], shoulder2[1] - shoulder1[1])
                # 计算单位向量
                magnitude = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
                unit_vector = (direction_vector[0] / magnitude, direction_vector[1] / magnitude)
                # 扩大肩部关键点距离10次
                while 10 - distance:
                    # 计算新的点坐标
                    shoulder1_copy = (
                        int(shoulder1[0] - distance * unit_vector[0]), int(shoulder1[1] - distance * unit_vector[1]))
                    shoulder2_copy = (
                        int(shoulder2[0] + distance * unit_vector[0]), int(shoulder2[1] + distance * unit_vector[1]))
                    dis1, coordinate1 = self.pyrs.get_3d_camera_coordinate(shoulder1_copy)
                    dis2, coordinate2 = self.pyrs.get_3d_camera_coordinate(shoulder2_copy)
                    if not coordinate1 or not coordinate2:
                        distance += 1
                    elif coordinate1 and coordinate2:
                        shoulder1x, shoulder1y, shoulder1z = round(coordinate1[0], 3), round(coordinate1[1], 3), round(
                            coordinate1[2], 3)
                        shoulder2x, shoulder2y, shoulder2z = round(coordinate2[0], 3), round(coordinate2[1], 3), round(
                            coordinate2[2], 3)
                        # 计算双肩的距离
                        shoulder_distance = (((shoulder1x - shoulder2x) ** 2) + ((shoulder1z - shoulder2z) ** 2)) ** 0.5
                        if min_shoulder_dis <= shoulder_distance <= max_shoulder_dis:
                            # print(
                            #     f'双肩的距离扩大{distance}次后 已经变得合理了:{round(shoulder_distance, 3)}m')
                            x1, x2, y1, y2 = shoulder1x, shoulder2x, shoulder1z, shoulder2z
                            # point0 = np.array([(x2 + x1) / 2, (y2 + y1) / 2])  # 双肩的中点
                            normal_vector = -np.array([y1 - y2, x2 - x1])  # 双肩的法向量
                            # 向量单位化
                            # normal_vector /= (((normal_vector[0] ** 2) + (normal_vector[1] ** 2)) ** 0.5)
                            # 判断单位向量的方向 先选取朝向向相机的那个
                            # normal_vector = normal_vector if np.linalg.norm(point0 + normal_vector) < np.linalg.norm(
                            #     point0 + normal_vector) else -normal_vector
                            # # 如果有头部信息 朝向不变 没有头部信息则改变向量方向
                            # if turn:
                            #     normal_vector = -normal_vector
                            angle_degrees = np.arctan2(normal_vector[1], normal_vector[0])
                            # angle_degrees = np.degrees(orientation)
                            break
                        elif shoulder_distance > max_shoulder_dis:
                            distance = 10
                        else:
                            distance += 1
                    # print(distance - 1, f'{round(shoulder_distance, 3)}m')
                if distance == 10:
                    # print(f'双肩距离扩大了10次任然不合理:{round(shoulder_distance, 3)}m 采用另一种方法测量')
                    # if shoulder1z < shoulder2z:
                    #     shoulder, left = shoulder1, True
                    # else:
                    #     shoulder, left = shoulder2, False
                    shoulder = shoulder1 if shoulder1z < shoulder2z else shoulder2
                    # print(
                    #     f'shoulder1xyz:{shoulder1x, shoulder1y, shoulder1z}, shoulder2xyz:{shoulder2x, shoulder2y, shoulder2z}')
                    # 如果双肩距离不正常是因为相机没有获取到深度信息 (没有深度信息的肩部点为(0.0, 0.0, 0.0))
                    # 那就根据(左、右)肩与头的关系来判断角度
                    if (not shoulder2x and not shoulder2y and not shoulder2z) or (
                            not shoulder1x and not shoulder1y and not shoulder1z):
                        # 左肩为0且有头部信息
                        if shoulder1x == 0 and not_empty_head:
                            # 右肩远
                            if not_empty_head[0][2] < shoulder2z:
                                return 0
                            # 右肩近
                            elif not_empty_head[0][2] > shoulder2z:
                                return 180
                        # 右肩为0且有头部信息
                        elif shoulder2x == 0 and not_empty_head:
                            # 左肩远
                            if not_empty_head[0][2] < shoulder1z:
                                return 180
                            # 左肩近
                            elif not_empty_head[0][2] > shoulder1z:
                                return 0
                        return None
                    cv2.circle(frame, (int(shoulder[0]), int(shoulder[1])), 10, (255, 100, 20), -1)
                    # 肩都有深度信息 但肩距不对 则用头部信息和离相机更近的肩判断方位
                    angle_degrees = self.shoulder_head(shoulder, not_empty_head, original_head)
                    return angle_degrees
        # 如果双肩中有一个点不存在或都不存在
        elif shoulder1[0]:
            cv2.circle(frame, (int(shoulder1[0]), int(shoulder1[1])), 10, (255, 100, 20), -1)
            angle_degrees = self.shoulder_head(shoulder1, not_empty_head, original_head)
        elif shoulder2[0]:
            cv2.circle(frame, (int(shoulder2[0]), int(shoulder2[1])), 10, (255, 100, 20), -1)
            angle_degrees = self.shoulder_head(shoulder2, not_empty_head, original_head)
        else:
            angle_degrees = None
        return angle_degrees

    def contral_mechanical_arm(self, type=b'a'):
        if self.ser.isOpen():
            print("打开串口成功, 串口号: %s" % self.ser.name)
            n = 5
            while n:
                self.ser.write(type)
                n -= 1
            print(type)

    def Close_to_person(self, X, Z, min_dis, speed=0.3478, camera_to_robot=0.66):
        # self.msg.linear.x = speed * (Z - min_dis + 0.1)
        if 3 < Z:
            self.msg.linear.x = 0.3
        else:
            self.msg.linear.x = speed * (Z - min_dis + 0.1)
        self.msg.angular.z = (np.arctan2(-X, camera_to_robot + Z) / 1.5)
        if Z <= min_dis:
            self.arrive_target_distance += 1
            # print(self.arrive_target_distance)
            if self.arrive_target_distance > 5:
                for _ in range(2):
                    self.msg.linear.x = 0
                    self.msg.angular.z = 0
                    self.pub.publish(self.msg)
                    self.rate.sleep()
                self.arrive_target_distance = 0
                return True
        else:
            self.arrive_target_distance = 0
        self.pub.publish(self.msg)
        # print('速度x:', self.msg.linear.x, 'z:', self.msg.angular.z)
        self.rate.sleep()

    def go_to_another_point(self, point):
        self.goal.target_pose.header.frame_id = "map"
        self.goal.target_pose.header.stamp = rospy.Time.now()
        self.goal.target_pose.pose.position.x = point[0]
        self.goal.target_pose.pose.position.y = point[1]
        self.goal.target_pose.pose.orientation.x = point[2]
        self.goal.target_pose.pose.orientation.y = point[3]
        self.goal.target_pose.pose.orientation.z = point[4]
        self.goal.target_pose.pose.orientation.w = point[5]
        self.client.send_goal(self.goal)
        print("waiting for move to point action...")
        while True:  # 在未到达定点时，相机一直处于阻塞状态
            wait = self.client.wait_for_result(rospy.Duration.from_sec(10))
            if wait:
                return None

    def val_point(self, avg_point, quaternion_lst, angle_degrees_lst):
        distance1_lst = []
        # 计算出各个最终目标导航点之间的欧氏距离
        for i in avg_point:
            distance = 0
            for j in avg_point:
                distance += np.linalg.norm([i[0] - j[0], i[1] - j[1]])
            distance1_lst.append([distance])
        # print('distance1_lst:', distance1_lst, '\n')
        # print('quaternion_lst:', quaternion_lst, '\n')

        # distance1_lst, quaternion_lst = [[i] for i in distance1_lst], [[i] for i in quaternion_lst]

        dbscan1 = DBSCAN(eps=3, min_samples=10)
        clusters1 = dbscan1.fit_predict(distance1_lst)
        outliers1 = np.where(clusters1 == -1)
        outliers1 = list(outliers1[0])

        dbscan2 = DBSCAN(eps=6, min_samples=10)
        clusters2 = dbscan2.fit_predict(quaternion_lst)
        outliers2 = np.where(clusters2 == -1)
        outliers2 = list(outliers2[0])

        dbscan3 = DBSCAN(eps=6, min_samples=10)
        clusters3 = dbscan3.fit_predict(angle_degrees_lst)
        outliers3 = np.where(clusters3 == -1)
        outliers3 = list(outliers3[0])

        avg_point = [item for index, item in enumerate(avg_point) if index not in outliers1]

        quaternion_lst = [item[0] for index, item in enumerate(quaternion_lst) if index not in outliers2]

        angle_degrees_lst = [item[0] for index, item in enumerate(angle_degrees_lst) if index not in outliers3]

        if bool(quaternion_lst) and bool(avg_point) and bool(angle_degrees_lst):
            # 将最终角度取平均值之后再转化为四元数中的后两位(前两位用不到 因为我们的车只能绕yaw轴旋转) 最终目标点也取均值
            z, w = self.euler2quaternion(np.mean(quaternion_lst))
            goal_point = [np.mean(np.array(avg_point)[:, 0]), np.mean(np.array(avg_point)[:, 1])]

            # z, w = np.mean(np.array(quaternion_lst)[:, 0]), np.mean(np.array(quaternion_lst)[:, 1])
            # print('goal_point:', goal_point, self.quaternion2euler([0.0, 0.0, z, w]), '\n')
            pose = [goal_point[0], goal_point[1], 0.0, 0.0, z, w]
            angle_degrees_lst = np.mean(angle_degrees_lst) % 360
            angle_degrees_lst = abs(360 - angle_degrees_lst) if angle_degrees_lst > 180 else angle_degrees_lst
            return pose, 180 - angle_degrees_lst
        else:
            return None, None

    def calculate_point(self, distance_far_away_person=3.0, camera_to_robot=0.66):
        avg_point, quaternion_lst, angle_degrees_lst = [], [], []
        # 获取机器人当前的x y z w(z w是四元数) 和欧拉角(范围是0-360度 以车最开始的方向↑为0 逆时针为正)
        # 放到循环外面是因为机器人在原地不动时 获取的位姿是一样的
        robot_x, robot_y, z, w, robot_euler = self.getIniPose()
        # date_input = datetime.datetime.now().strftime('_%m_%d_%H_%M_%S')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # output_path = f"/home/gardenia/zhongji_ws/vision/src/liu/behavioral_photo/output{date_input}.avi"
        # out = cv2.VideoWriter(output_path, fourcc, frameSize=(640, 480), fps=20)
        for _ in range(50):
            frame, depth_colormap = self.pyrs.get_img()
            frame, people_count, shoulder_head_counts, _ = self.BodyDetector.inference(frame)
            if not people_count:
                continue
            for index, i in enumerate(people_count):
                person_distance, coordinate = self.pyrs.get_3d_camera_coordinate(i[1])
                # 找到人的次数大于5次且人的距离不大于5m
                if not coordinate or person_distance > 5:
                    # not_found_people += 1
                    continue
                # 这个X Y Z是以人为世界坐标系 相机在此坐标系中的三维坐标 同样的可以将相机当作坐标原点
                X, Y, Z = round(coordinate[0], 4), round(coordinate[1], 4), round(coordinate[2], 4)
                # 这里算的角度是人在以机器人底盘为原点的角度 以正常坐标系的x轴(→)为0弧度 逆时针为正
                person_to_robot_angle_degrees = np.arctan2(camera_to_robot + Z, X)
                # 将弧度转化为角度 然后减去90 这样就将0度的线变成了y轴正方向(↑) 与机器人的方向表达保持相同
                person_to_robot_angle_degrees = np.degrees(person_to_robot_angle_degrees) - 90
                # print('person_to_robot_angle_degrees:', person_to_robot_angle_degrees)
                # 计算人到机器人底盘的二维距离(就是从地图上来看两点之间的距离) 因为虽然有一个旋转机器人使人位于屏幕中心的过程 但是往往有误差 误差太多了精度就很低
                person_to_robot_distance = (X ** 2 + (Z + camera_to_robot) ** 2) ** 0.5
                # print('person_to_robot_distance:', person_to_robot_distance)
                # 这里计算以机器人的地图为原坐标系 将原坐标系平移到相机的位置下 人在此坐标系下的坐标
                # (机器人的地图坐标系是左边←为y轴正方向 上面↑为x轴正方向)
                person_to_robot_in_o_x, person_to_robot_in_o_y = (
                    # 将人相对于机器人底盘的角度加上机器人当前的角度(已经将二者的起点、正负等调成一致的情况下) 然后再转化为弧度 再计算
                    person_to_robot_distance * np.cos(np.radians(robot_euler + person_to_robot_angle_degrees)),
                    person_to_robot_distance * np.sin(np.radians(robot_euler + person_to_robot_angle_degrees))
                )
                # print('person_to_robot_in_o_x, person_to_robot_in_o_y:', person_to_robot_in_o_x, person_to_robot_in_o_y)
                # 鼻子、眼睛、眼睛、肩膀、肩膀 格式为[,] [,] [,] [,] [,]
                nose, eye1, eye2, shoulder1, shoulder2 = (
                    shoulder_head_counts[0][0], shoulder_head_counts[0][1],
                    shoulder_head_counts[0][2], shoulder_head_counts[0][5],
                    shoulder_head_counts[0][6]
                )
                # 通过一系列数学判断获取人的朝向 没获取成功则angle_degrees=None 即舍去当前检测结果
                angle_degrees = self.determine_personnel_orientation(nose, eye1, eye2, shoulder1, shoulder2, frame)
                if angle_degrees:
                    # 得到的角度是以x轴负方向←为0度顺时针为负 逆时针为正 范围是-pi到pi
                    # print('前angle_degrees:', np.degrees(angle_degrees), '度数', angle_degrees, '弧度')
                    if angle_degrees > self.PI:
                        angle_degrees -= 3 * self.PI
                    elif angle_degrees >= -0.5 * self.PI:
                        angle_degrees += 0.5 * self.PI
                    elif angle_degrees < -0.5 * self.PI:
                        angle_degrees += 2.5 * self.PI
                    # 通过上面的if判断 可以将角度的起始线调整为以y轴正方向↑为0弧度 逆时针为正
                    angle_degrees = np.degrees(angle_degrees)
                    # print('后angle_degrees:', angle_degrees)
                    # 由于人在相机中的角度与机器人在地图上的角度的标准已经调整至一致 所以人在地图中的角度可以由简单的相加得到
                    person_angle_in_o_degrees = angle_degrees + robot_euler
                    # print('person_angle_in_o_degrees:', person_angle_in_o_degrees)
                    # x2, y2是人在地图中的坐标位置 是以左边←为y轴正方向 上方↑为x轴正方向
                    x2, y2 = robot_x + person_to_robot_in_o_x, robot_y + person_to_robot_in_o_y
                    # print('人在坐标系中的位置x2:', x2, 'y2:', y2, '\n')
                    # 通过人在地图中的坐标和人在地图中的角度来计算最终的目标导航点 *distance_far_away_person是里计算出来的最终目标导航点离人的距离
                    x3, y3 = (
                        x2 + np.cos(np.radians(person_angle_in_o_degrees)) * distance_far_away_person,
                        y2 + np.sin(np.radians(person_angle_in_o_degrees)) * distance_far_away_person
                    )
                    if person_angle_in_o_degrees > 10000:
                        continue
                    # 因为机器人导航到定位点之后还要调整方位 所以还要计算机器人的方位 因为人和机器已经在一条直线上了 所以二者相差180度
                    # quaternion_lst.append(person_angle_in_o_degrees + 180)
                    quaternion_lst.append([person_angle_in_o_degrees + 180])
                    avg_point.append([x3, y3])
                    angle_degrees_lst.append([angle_degrees])
            # out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                self.pyrs.pipeline.stop()
                exit(0)
        cv2.destroyAllWindows()
        # out.release()

        return avg_point, quaternion_lst, angle_degrees_lst

    def find_and_close_person(self, min_person_distance=5.5):
        need_speed_init, exit_flag, found_people, do_not_found_people = True, True, False, 0
        while exit_flag:
            person_distance_dict = {}
            try:
                color_image, _ = self.pyrs.get_img()
                frame, people_count, _, _ = self.BodyDetector.inference(color_image)
                if not people_count:
                    do_not_found_people += 1
                    if do_not_found_people % 20 == 0:
                        self.msg.linear.x = 0
                        self.msg.angular.z = -0.2
                        self.pub.publish(self.msg)
                        self.rate.sleep()
                    continue
                for index, i in enumerate(people_count):
                    # print('find_and_close_person i:', i[1])
                    person_distance, coordinate = self.pyrs.get_3d_camera_coordinate(i[1])
                    if not coordinate or person_distance > min_person_distance:
                        print(f"未获取到人的深度信息 或检测对象距离超过{min_person_distance}m")
                        continue
                    X, Y, Z = round(coordinate[0], 4), round(coordinate[1], 4), round(coordinate[2], 4)
                    person_distance_dict[index] = [person_distance, X, Z]
                if bool(person_distance_dict):
                    object_person_key = sorted(person_distance_dict, key=lambda k: person_distance_dict[k][0], reverse=False)[0]
                    object_person_value = person_distance_dict[object_person_key]
                    X, Z = object_person_value[1], object_person_value[2]
                    # 进行速度初始化 减轻车突然翘头的副作用
                    if need_speed_init:
                        self.initalization_speed()
                        need_speed_init = False
                    # 把控车到人的距离 一旦距离小于min_dis车就会停下 然后退出这个while循环
                    elif self.Close_to_person(X, Z, min_dis=2.2, speed=0.45):
                        exit_flag = False
                    # print('X:', X, "Z:", Z)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    self.pyrs.pipeline.stop()
                    exit(0)
            except Exception as e:
                print(e)
        cv2.destroyAllWindows()

    def navigate_to_fixed_point(self, distance_far_away_person=3.0):
        while True:
            person_distance_dict = {}
            try:
                avg_point, quaternion_lst, angle_degrees_lst = self.calculate_point(distance_far_away_person=distance_far_away_person)
                point, _ = self.val_point(avg_point, quaternion_lst, angle_degrees_lst)
                if point:
                    print('point:', point)
                    break
                else:
                    color_image, _ = self.pyrs.get_img()
                    frame, people_count, _, _ = self.BodyDetector.inference(color_image)
                    for index, i in enumerate(people_count):
                        central_point = ((i[0][0] + i[0][2]) // 2, (i[0][1] + i[0][3]) // 2)
                        person_distance, coordinate = self.pyrs.get_3d_camera_coordinate(central_point)
                        if not coordinate or person_distance > 4.5:
                            print("未获取到人的深度信息 或检测对象距离超过4.5m")
                            continue
                        X, Y, Z = round(coordinate[0], 4), round(coordinate[1], 4), round(coordinate[2], 4)
                        person_distance_dict[index] = [person_distance, X, Z]
                    if bool(person_distance_dict):
                        object_person_key = \
                        sorted(person_distance_dict, key=lambda k: person_distance_dict[k][0], reverse=False)[0]
                        object_person_value = person_distance_dict[object_person_key]
                        X, Z = object_person_value[1], object_person_value[2]
                        if Z < 1.5:
                            self.msg.linear.x = -0.1
                            self.msg.angular.z = 0
                            self.pub.publish(self.msg)
                            self.rate.sleep()
                        else:
                            self.msg.linear.x = 0.1
                            self.msg.angular.z = 0
                            self.pub.publish(self.msg)
                            self.rate.sleep()
            except Exception as e:
                print(e)
        print('座标点计算完毕')
        self.go_to_another_point(point)
        print('导航结束')
        for _ in range(3):
            try:
                avg_point, quaternion_lst, angle_degrees_lst = self.calculate_point(distance_far_away_person=distance_far_away_person)
                print('再次座标点计算完毕!!!')
                point, angle_degrees = self.val_point(avg_point, quaternion_lst, angle_degrees_lst)
                print('angle_degrees:', angle_degrees)
                break
            except Exception as e:
                point, angle_degrees = None, None
                print(e)
        return point, angle_degrees


    def correction_point(self, min_person_distance=5.5):
        start_time = time.time()
        n = 0
        while n < 20:
            person_distance_dict = {}
            try:
                frame, _ = self.pyrs.get_img()
                frame, people_count, _, _ = self.BodyDetector.inference(frame)
                n += 1
                # print('correction_point:', n)
                if not people_count:
                    if n % 10 == 0:
                        self.msg.linear.x = 0
                        self.msg.angular.z = -0.2
                        self.pub.publish(self.msg)
                        self.rate.sleep()
                    continue
                else:
                    for index, i in enumerate(people_count):
                        central_point = ((i[0][0] + i[0][2]) // 2, (i[0][1] + i[0][3]) // 2)
                        person_distance, coordinate = self.pyrs.get_3d_camera_coordinate(central_point)
                        if not coordinate or person_distance > min_person_distance:
                            print(f"未获取到人的深度信息 或检测对象距离超过{min_person_distance}m")
                            continue
                        X, Y, Z = round(coordinate[0], 4), round(coordinate[1], 4), round(coordinate[2], 4)
                        person_distance_dict[index] = [person_distance, X, Z]
                    if bool(person_distance_dict):
                        object_person_key = \
                        sorted(person_distance_dict, key=lambda k: person_distance_dict[k][0], reverse=False)[0]
                        object_person_value = person_distance_dict[object_person_key]
                        X, Z = object_person_value[1], object_person_value[2]
                        self.msg.linear.x = 0
                        self.msg.angular.z = (2 * np.arctan2(-X, 0.66 + Z))
                        self.pub.publish(self.msg)
                        self.rate.sleep()
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    self.pyrs.pipeline.stop()
                    exit(0)
            except Exception as e:
                print(e)
        cv2.destroyAllWindows()
        print('correction_point_time:', round(time.time() - start_time, 3))

    def fall_back(self, distance_robot_person=1.5):
        arrive_at_distance = 0
        global front_distance
        while True:
            person_distance_dict = {}
            try:
                color_image, _ = self.pyrs.get_img()
                frame, people_count, _, bone_frame = self.BodyDetector.inference(color_image)
                if not people_count:
                    self.msg.linear.x = 0
                    self.msg.angular.z = -0.2
                    self.pub.publish(self.msg)
                    self.rate.sleep()
                    continue
                for index, i in enumerate(people_count):
                    central_point = ((i[0][0] + i[0][2]) // 2, (i[0][1] + i[0][3]) // 2)
                    person_distance, coordinate = self.pyrs.get_3d_camera_coordinate(central_point)
                    if not coordinate:
                        print("未获取到人的深度信息")
                        continue
                    X, Y, Z = round(coordinate[0], 4), round(coordinate[1], 4), round(coordinate[2], 4)
                    person_distance_dict[index] = [person_distance, X, Z]
                if bool(person_distance_dict):
                    object_person_key = \
                    sorted(person_distance_dict, key=lambda k: person_distance_dict[k][0], reverse=False)[0]
                    object_person_value = person_distance_dict[object_person_key]
                    X, Z = object_person_value[1], object_person_value[2]
                    # distance = distance_queue.get()
                    distance = front_distance
                    print(f'person_distance:{object_person_value[0]}, distance:{distance}')
                    if object_person_value[0] > distance_robot_person or distance < 0.9:
                        # if distance < 1:
                        arrive_at_distance += 1
                        if arrive_at_distance > 5:
                            self.msg.linear.x = 0
                            self.msg.angular.z = 0
                            self.pub.publish(self.msg)
                            self.rate.sleep()
                            if distance < 0.9:
                                return False
                            else:
                                return True
                        elif 0.5 < object_person_value[0] < 1:
                            self.msg.linear.x = -0.15
                            self.msg.angular.z = (np.arctan2(-X, 0.66 + Z) / 1.5)
                            self.pub.publish(self.msg)
                            self.rate.sleep()
                        else:
                            self.msg.linear.x = -0.15
                            self.msg.angular.z = (np.arctan2(-X, 0.66 + Z) * 1.5)
                            self.pub.publish(self.msg)
                            self.rate.sleep()
                    elif 0.5 < object_person_value[0] < 1:
                        self.msg.linear.x = -0.15
                        self.msg.angular.z = (np.arctan2(-X, 0.66 + Z) / 1.5)
                        self.pub.publish(self.msg)
                        self.rate.sleep()
                    else:
                        self.msg.linear.x = -0.15
                        self.msg.angular.z = (np.arctan2(-X, 0.66 + Z) * 1.5)
                        self.pub.publish(self.msg)
                        self.rate.sleep()
                cv2.imshow('frame', frame)
                # cv2.imshow('bone_frame', bone_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    self.pyrs.pipeline.stop()
                    exit(0)
            except Exception as e:
                print(e)
        cv2.destroyAllWindows()

    def face_detector(self):
        none_face = True
        none_right_face = True
        face_pattern = True
        identy_face_and_action = True
        identy_action = False
        # for _ in range(50):
        while identy_face_and_action:
            if self.get_face_database():
                try:
                    img_rd, depth_colormap = self.pyrs.get_img()
                    if face_pattern:
                        logging.debug("Frame %d starts", self.frame_cnt)
                        faces = detector(img_rd, 0)
                        self.draw_note(img_rd)
                        self.current_frame_face_feature_list = []
                        self.current_frame_face_cnt = 0
                        self.current_frame_face_name_position_list = []
                        self.current_frame_face_name_list = []

                        # 2. 检测到人脸 / Face detected in current frame
                        if len(faces) != 0:
                            none_face = True
                            # 3. 获取当前捕获到的图像的所有人脸的特征 / Compute the face descriptors for faces in current frame
                            for i in range(len(faces)):
                                shape = predictor(img_rd, faces[i])
                                self.current_frame_face_feature_list.append(
                                    face_reco_model.compute_face_descriptor(img_rd, shape))
                            # 4. 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                            for k in range(len(faces)):
                                logging.debug("For face %d in camera:", k + 1)
                                # 先默认所有人不认识，是 unknown / Set the default names of faces with "unknown"
                                self.current_frame_face_name_list.append("unknown")

                                # 每个捕获人脸的名字坐标 / Positions of faces captured
                                self.current_frame_face_name_position_list.append(tuple(
                                    [faces[k].left(),
                                     int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                                # 5. 对于某张人脸，遍历所有存储的人脸特征
                                # For every faces detected, compare the faces in the database
                                current_frame_e_distance_list = []
                                for i in range(len(self.face_feature_known_list)):
                                    # 如果 person_X 数据不为空
                                    if str(self.face_feature_known_list[i][0]) != '0.0':
                                        e_distance_tmp = self.return_euclidean_distance(
                                            self.current_frame_face_feature_list[k],
                                            self.face_feature_known_list[i])
                                        logging.debug("  With person %s, the e-distance is %f", str(i + 1),
                                                      e_distance_tmp)
                                        current_frame_e_distance_list.append(e_distance_tmp)
                                    else:
                                        # 空数据 person_X
                                        current_frame_e_distance_list.append(999999999)
                                # 6. 寻找出最小的欧式距离匹配 / Find the one with minimum e-distance
                                similar_person_num = current_frame_e_distance_list.index(
                                    min(current_frame_e_distance_list))
                                logging.debug("Minimum e-distance with %s: %f",
                                              self.face_name_known_list[similar_person_num],
                                              min(current_frame_e_distance_list))

                                if min(current_frame_e_distance_list) < 0.4:
                                    self.current_frame_face_name_list[k] = self.face_name_known_list[
                                        similar_person_num]
                                    logging.debug("Face recognition result: %s",
                                                  self.face_name_known_list[similar_person_num])
                                else:
                                    logging.debug("Face recognition result: Unknown person")
                                logging.debug("\n")

                                # 矩形框 / Draw rectangle
                                for kk, d in enumerate(faces):
                                    # 绘制矩形框
                                    cv2.rectangle(img_rd, tuple([d.left(), d.top()]),
                                                  tuple([d.right(), d.bottom()]),
                                                  (255, 255, 255), 2)

                            self.current_frame_face_cnt = len(faces)

                            # 7. 在这里更改显示的人名 / Modify name if needed
                            # self.show_chinese_name()

                            # 8. 写名字 / Draw name
                            img_with_name = self.draw_name(img_rd)
                            count_unknown = 0
                            for name in self.current_frame_face_name_list:
                                print(name)
                                if name == 'unknown':
                                    count_unknown += 1
                                else:
                                    # name = name.encode('utf-16').decode('utf-16')
                                    playsound('/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/name/' + name + ".wav")
                                    print('播发/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/name/{}.wav'.format(name))
                                    identy_face_and_action = False

                                if count_unknown == len(self.current_frame_face_name_list):
                                    if none_right_face:
                                        none_right_face_begin_time = time.time()
                                        none_right_face = False
                                    if time.time() > none_right_face_begin_time + 10:
                                        print('flase_re_people')
                                        identy_face_and_action = False

                        else:
                            none_right_face = True
                            if none_face:
                                none_face_begin_time = time.time()
                                none_face = False
                            if time.time() > none_face_begin_time + 3:
                                identy_face_and_action = False

                            img_with_name = img_rd

                    cv2.imshow('xxx', img_with_name)
                    # 9. 更新 FPS / Update stream FPS
                    self.update_fps()
                    # cv2.waitKey(0)
                    logging.debug("Frame ends\n\n")
                    cv2.waitKey(1)
                except Exception as e:
                    print(e)

    def action_detector(self):
        self.contral_mechanical_arm(b'e')
        time.sleep(1.5)
        open_camera = True
        self.hand_switch = True
        self.next_action = False
        self.action_re_switch = False
        success = 0
        play_wav_file('/home/gardenia/zhongji_ws/vision/src/tt01/action/next_action.wav')
        time.sleep(5.0)
        play_wav_file('/home/gardenia/zhongji_ws/vision/src/tt01/action/begin.wav')
        try:
            last_key = '挥手'
            frame_num = 0
            with self.My_pose_point.mp_pose.Pose(
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5) as pose:
                while open_camera:
                    image, depth_colormap = self.pyrs.get_img()
                    image_h, image_w = image.shape[:2]
                    fps_time = time.time()
                    frame_num += 1
                    if self.hand_switch:
                        body_point = []
                        results = self.model(image)
                        keypoints = results[0].keypoints
                        print(keypoints)
                        if keypoints is not None:
                            keypoints_cpu = keypoints.cpu()

                            for keypoint in keypoints_cpu:
                                # print(keypoints.xy.shape)

                                for xy in keypoint.xy[0]:
                                    x1, y1 = map(int, xy)
                                    body_point.append((x1, y1))
                                    cv2.circle(image, (x1, y1), 5, (255, 255, 0), -1)
                            if len(body_point) != 0:
                                right_eye = body_point[2]
                                left_eye = body_point[1]
                                left_hand = body_point[9]
                                right_hand = body_point[10]
                                left_elbow = body_point[7]
                                right_elbow = body_point[8]
                                # 两条线段的端点坐标表示为((x1, y1), (x2, y2))
                                segment1 = (left_hand, left_elbow)
                                segment2 = (right_hand, right_elbow)

                                count = 0
                                hand = False
                                call_count = 0
                                call = False
                                if left_hand[0] != 0 and left_hand[1] != 0:
                                    if left_hand[1] < left_eye[1]:
                                        hand = True

                                        self.left_track.append(
                                            (float(left_hand[0]), float(left_hand[1])))  # x, y center point
                                        if len(self.left_track) > 15:  # retain 90 tracks for 90 frames
                                            self.left_track.pop(0)

                                        # Draw the tracking lines
                                        points = np.hstack(self.left_track).astype(np.int32).reshape((-1, 1, 2))
                                        cv2.polylines(image, [points], isClosed=False, color=(230, 230, 230),
                                                      thickness=10)

                                        # 计算多边形的周长
                                        perimeter = 0
                                        for i in range(len(points) - 1):
                                            perimeter += np.linalg.norm(points[i] - points[i + 1])

                                        # 计算多边形最后一个顶点到第一个顶点的距离
                                        perimeter += np.linalg.norm(points[-1] - points[0])

                                        if perimeter > 150:
                                            count += 1
                                        print("左手周长为:", perimeter)

                                    elif left_eye[1] < left_hand[1] < left_elbow[1]:
                                        call_count += 1
                                        call = True

                                if right_hand[0] != 0 and right_hand[1] != 0:
                                    if right_hand[1] < right_eye[1]:
                                        hand = True
                                        self.right_track.append(
                                            (float(right_hand[0]), float(right_hand[1])))  # x, y center point
                                        if len(self.right_track) > 15:  # retain 90 tracks for 90 frames
                                            self.right_track.pop(0)

                                        # Draw the tracking lines
                                        points = np.hstack(self.right_track).astype(np.int32).reshape((-1, 1, 2))
                                        cv2.polylines(image, [points], isClosed=False, color=(0, 230, 230),
                                                      thickness=10)

                                        # 计算多边形的周长
                                        perimeter = 0
                                        for i in range(len(points) - 1):
                                            perimeter += np.linalg.norm(points[i] - points[i + 1])

                                        # 计算多边形最后一个顶点到第一个顶点的距离
                                        perimeter += np.linalg.norm(points[-1] - points[0])

                                        if perimeter > 150:
                                            count += 1
                                        print("右手手周长为:", perimeter)

                                    elif right_eye[1] < right_hand[1] < right_elbow[1]:
                                        call_count += 1
                                        call = True

                                if left_hand[0] != 0 and left_hand[1] != 0 and right_hand[0] != 0 and right_hand[
                                    1] != 0:
                                    cv2.line(image, left_hand, left_elbow, (255, 255, 255), 2)
                                    cv2.line(image, right_hand, right_elbow, (255, 255, 255), 2)
                                    if intersect(segment1, segment2):
                                        print("线段相交")
                                        if self.success_time_begin_4_switch:
                                            success_begin_4_time = time.time()
                                            print("time: {}".format(success_begin_4_time))
                                            self.success_time_begin_4_switch = False
                                        cv2.putText(image, 'chashou', (20, 40), cv2.FONT_ITALIC, 0.8,
                                                    (0, 255, 0), 1,
                                                    cv2.LINE_AA)
                                        if time.time() > success_begin_4_time + 2:
                                            success += 1
                                            print('叉手')
                                            audio_thread = threading.Thread(target=play_wav_file,
                                                                            args=(
                                                                                '/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/action/chaoshou.wav',))
                                            audio_thread.start()
                                            self.hand_switch = False
                                            self.next_action = True
                                            self.success_time_begin_0_switch = True
                                            self.success_time_begin_1_switch = True
                                            self.success_time_begin_2_switch = True
                                            self.success_time_begin_3_switch = True
                                            self.success_time_begin_4_switch = True
                                            self.next = True

                                if hand:
                                    self.false_time_begin_switch = True
                                    if count == 0:
                                        if self.success_time_begin_0_switch:
                                            success_begin_0_time = time.time()
                                            print("time: {}".format(success_begin_0_time))
                                            self.success_time_begin_0_switch = False
                                        cv2.putText(image, 'hand up', (20, 40), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 1,
                                                    cv2.LINE_AA)
                                        if time.time() > success_begin_0_time + 2:
                                            success += 1
                                            print('hand up')
                                            audio_thread = threading.Thread(target=play_wav_file,
                                                                            args=(
                                                                                '/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/action/hand_up.wav',))
                                            audio_thread.start()
                                            self.hand_switch = False
                                            self.next_action = True
                                            self.success_time_begin_0_switch = True  # 举手
                                            self.success_time_begin_1_switch = True  # 挥手
                                            self.success_time_begin_2_switch = True  # 会双手
                                            self.success_time_begin_3_switch = True  # 打电话
                                            self.success_time_begin_4_switch = True  # 叉手
                                            self.next = True

                                    elif count == 1:
                                        if self.success_time_begin_1_switch:
                                            success_begin_1_time = time.time()
                                            self.success_time_begin_1_switch = False
                                        cv2.putText(image, 'wave', (20, 40), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 1,
                                                    cv2.LINE_AA)
                                        if time.time() > success_begin_1_time + 2:
                                            success += 1
                                            print('wave')
                                            audio_thread = threading.Thread(target=play_wav_file, args=(
                                                '/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/action/wave.wav',))
                                            audio_thread.start()
                                            self.hand_switch = False
                                            self.next_action = True
                                            self.success_time_begin_0_switch = True  # 举手
                                            self.success_time_begin_1_switch = True  # 挥手
                                            self.success_time_begin_2_switch = True  # 会双手
                                            self.success_time_begin_3_switch = True  # 打电话
                                            self.success_time_begin_4_switch = True  # 叉手
                                            self.next = True

                                    elif count == 2:
                                        if self.success_time_begin_2_switch:
                                            success_begin_2_time = time.time()
                                            self.success_time_begin_2_switch = False
                                        cv2.putText(image, 'wave two hand', (20, 40), cv2.FONT_ITALIC, 0.8, (0, 255, 0),
                                                    1,
                                                    cv2.LINE_AA)
                                        if time.time() > success_begin_2_time + 2:
                                            success += 1
                                            print('wave two hand')
                                            audio_thread = threading.Thread(target=play_wav_file,
                                                                            args=(
                                                                                '/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/action/wave_two_hand.wav',))
                                            audio_thread.start()
                                            self.hand_switch = False
                                            self.next_action = True
                                            self.success_time_begin_0_switch = True  # 举手
                                            self.success_time_begin_1_switch = True  # 挥手
                                            self.success_time_begin_2_switch = True  # 会双手
                                            self.success_time_begin_3_switch = True  # 打电话
                                            self.success_time_begin_4_switch = True  # 叉手
                                            self.next = True

                                elif call:
                                    self.false_time_begin_switch = True
                                    if call_count == 1:
                                        if self.success_time_begin_3_switch:
                                            success_begin_3_time = time.time()
                                            print("time: {}".format(success_begin_3_time))
                                            self.success_time_begin_3_switch = False
                                        cv2.putText(image, 'call', (20, 40), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 1,
                                                    cv2.LINE_AA)
                                        if time.time() > success_begin_3_time + 2:
                                            success += 1
                                            print('call')
                                            audio_thread = threading.Thread(target=play_wav_file, args=(
                                                '/home/gardenia/zhongji_ws/vision/src/zhongji/identiy_people_and_action/action/call.wav',))
                                            audio_thread.start()
                                            self.hand_switch = False
                                            self.next_action = True
                                            self.success_time_begin_0_switch = True
                                            self.success_time_begin_1_switch = True
                                            self.success_time_begin_2_switch = True
                                            self.success_time_begin_3_switch = True
                                            self.next = True

                                else:
                                    self.success_time_begin_0_switch = True  # 举手
                                    self.success_time_begin_1_switch = True  # 挥手
                                    self.success_time_begin_2_switch = True  # 会双手
                                    self.success_time_begin_3_switch = True  # 打电话
                                    self.success_time_begin_4_switch = True  # 叉手
                                    if self.false_time_begin_switch:
                                        false_begin_time = time.time()
                                        self.false_time_begin_switch = False
                                    if time.time() > false_begin_time + 2:
                                        self.false_time_begin_switch = True
                                        self.action_re_switch = True
                                        self.hand_switch = False

                    if self.action_re_switch:
                        # 提高性能 使数组变为只读，防止对其数据进行修改。这在确保图像数据的完整性并避免意外更改时非常有用。
                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = pose.process(image)

                        if not results.pose_landmarks:
                            continue

                            # 识别骨骼点
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        landmarks = results.pose_landmarks.landmark
                        joints = np.array([[landmarks[joint].x * image_w,
                                            landmarks[joint].y * image_h,
                                            landmarks[joint].visibility]
                                           for joint in self.My_pose_point.KEY_JOINTS])
                        # 人体框
                        box_l, box_r = int(joints[:, 0].min()) - 50, int(joints[:, 0].max()) + 50
                        box_t, box_b = int(joints[:, 1].min()) - 100, int(joints[:, 1].max()) + 100

                        self.joints_list.append(joints)
                        # joints_lsit 为连续30帧的关见点
                        result = self.actin.action_re(self.joints_list, image_w, image_h)
                        if result is None:
                            print("Action recognition failed.")
                            continue
                        action_name, action = result
                        this_key = action
                        if this_key != last_key:
                            begin = time.time()
                        last_key = this_key
                        if time.time() > begin + 2:
                            success += 1
                            audio_thread = threading.Thread(target=play_wav_file, args=(
                                '/home/gardenia/zhongji_ws/vision/src/tt01/action/' + this_key + '.wav',))
                            audio_thread.start()
                            self.action_re_switch = False
                            self.next_action = True
                            self.next = True
                            last_key = '挥手'

                        image = self.My_pose_point.draw_skeleton(image, self.joints_list[-1])
                        image = cv2.rectangle(image, (box_l, box_t), (box_r, box_b), (255, 0, 0), 1)
                        image = cv2_add_chinese_text(self.word_path, image, f'当前状态：{action}',
                                                     (box_l + 10, box_t + 10),
                                                     (0, 255, 0), 40)
                        image = cv2.putText(image, f'FPS: {int(1.0 / (time.time() - fps_time))}',
                                            (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

                    if success >= 2:
                        open_camera = False
                        self.next_action = False

                    if self.next_action:
                        if self.wait_switch:
                            print('请开始下一个动作')

                            begin_time = time.time()
                            self.wait_switch = False
                        if time.time() > begin_time + 3:
                            if self.next:
                                audio_thread = threading.Thread(target=play_wav_file, args=(
                                    '/home/gardenia/zhongji_ws/vision/src/tt01/action/next_action.wav',))
                                audio_thread.start()
                                self.next = False
                            if time.time() > begin_time + 8:
                                self.next_action = False
                                self.wait_switch = True
                                print('开始识别下一个动作')
                                self.joints_list.clear()
                                self.left_track.clear()
                                self.right_track.clear()
                                audio_thread = threading.Thread(target=play_audio, args=(
                                    '/home/gardenia/zhongji_ws/vision/src/tt01/action/begin_identiy.wav',))
                                audio_thread.start()

                    status = check_playback_status()
                    if status:
                        self.hand_switch = True
                        play_finished.clear()

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    cv2.waitKey(1)
                    cv2.imshow('MediaPipe Pose', image)

        except Exception as e:
            print(e)

    def Control_distance(self, distance_far_away_person=3.0, min_person_distance=5.5):
        self.correction_point(min_person_distance=min_person_distance)
        self.find_and_close_person(min_person_distance=min_person_distance)
        point, angle_degrees = self.navigate_to_fixed_point(distance_far_away_person)
        self.correction_point(min_person_distance=min_person_distance)
        if point:
            print('angle_degrees:', angle_degrees)
            if angle_degrees > 20:
                print('启动了第二次导航')
                self.go_to_another_point(point)
                self.correction_point()
        should_fall_back = self.fall_back(distance_robot_person=0.8)
        self.correction_point(min_person_distance=min_person_distance)
        self.face_detector()
        if should_fall_back:
            should_fall_back = self.fall_back(distance_robot_person=1.5)
            self.correction_point(min_person_distance=min_person_distance)
        self.action_detector()

    # 一套流程是 找人 没找到就旋转 找到人之后靠近 导航到人的正面并距离人一定位置 行为识别 抓取物品
    def A_set_of_processes(self, req, point, biggest_find_person_count, room, min_person_distance=5.5):
        if not self.h_mechanical_arm:
            self.contral_mechanical_arm(b'h')
            self.h_mechanical_arm = True
            time.sleep(2)

        if room == 1:
            self.go_into_or_out_room(self.room1_in)
        elif room == 2:
            self.go_into_or_out_room(self.room2_in)
        self.go_to_another_point(point)
        angular_count, turn_right = 0, True
        # 没有找到人的次数 找到人的次数 退出找人的标志 是否需要进行行为识别
        do_not_found_people, found_people, exit_find_person, should_determine_behaviour = 0, 0, True, True
        while exit_find_person:
            person_distance_dict = {}
            try:
                color_image, _ = self.pyrs.get_img()
                frame, people_count, _, _ = self.BodyDetector.inference(color_image)
                # 未找到人则旋转
                if not people_count:
                    found_people = 0
                    do_not_found_people += 1
                    if do_not_found_people % 20 == 0:
                        print(f'因为未检测到人旋转了{do_not_found_people // 20}次')
                        z_angular = 0.3 if do_not_found_people <= 40 else 0.2
                        self.msg.angular.z = -z_angular if turn_right else z_angular
                        self.msg.linear.x = 0
                        self.pub.publish(self.msg)
                        self.rate.sleep()
                    elif do_not_found_people > biggest_find_person_count:
                        print('angular_count:', angular_count)
                        do_not_found_people = 0
                        angular_count += 1
                        time.sleep(1)
                        self.msg.angular.z = 0
                        self.msg.linear.x = 0
                        self.pub.publish(self.msg)
                        self.rate.sleep()
                        turn_right = False if turn_right else True
                        if angular_count >= 2:
                            exit_find_person, should_determine_behaviour = False, False
                else:
                    for index, i in enumerate(people_count):
                        central_point = ((i[0][0] + i[0][2]) // 2, (i[0][1] + i[0][3]) // 2)
                        person_distance, coordinate = self.pyrs.get_3d_camera_coordinate(central_point)
                        if not coordinate or person_distance > min_person_distance:
                            continue
                        X, Y, Z = round(coordinate[0], 4), round(coordinate[1], 4), round(coordinate[2], 4)
                        person_distance_dict[index] = [person_distance, X, Z]
                        print('person_distance:', person_distance)
                    if bool(person_distance_dict):
                        found_people += 1
                        if found_people >= 2:
                            self.Control_distance(distance_far_away_person=1.5, min_person_distance=min_person_distance)
                            print('over fixed point!!!!!!', round(time.time() - self.start_time))
                            exit_find_person = False
                    else:
                        found_people = 0
                        do_not_found_people += 1
                        # 连续30次没找到人则认为该房间没有人 直接清扫垃圾
                        if do_not_found_people % 20 == 0:
                            # print(f'因为未检测到人旋转了{do_not_found_people % 20}次')
                            z_angular = 0.3 if do_not_found_people <= 40 else 0.2
                            self.msg.angular.z = -z_angular if turn_right else z_angular
                            self.msg.linear.x = 0
                            self.pub.publish(self.msg)
                            self.rate.sleep()
                        elif do_not_found_people > biggest_find_person_count:
                            do_not_found_people = 0
                            angular_count += 1
                            time.sleep(1)
                            self.msg.angular.z = 0
                            self.msg.linear.x = 0
                            self.pub.publish(self.msg)
                            self.rate.sleep()
                            turn_right = False if turn_right else True
                            if angular_count >= 2:
                                exit_find_person, should_determine_behaviour = False, False
                # print('do_not_found_people:', do_not_found_people)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cv2.destroyAllWindows()
                    self.pyrs.pipeline.stop()
                    exit(0)
            except Exception as e:
                print(e)
        if room == 1:
            self.go_into_or_out_room(self.room1_out)
        elif room == 2:
            self.go_into_or_out_room(self.room2_out)
        cv2.destroyAllWindows()
        return should_determine_behaviour


    def start(self, req, type=1):
        total_person_count = 0
        self.room1_in = [7.546, -5.35, 0, 0, 0.678, 0.734]
        self.room1_out = [7.458, -3.68, 0, 0, -0.711, 0.703]
        self.room2_in = [4.37, -4.95, 0, 0, -0.73, 0.677]
        self.room2_out = [4.381, -6.73, 0, 0, 0.672, 0.74]
        if type == 1:
            room1 = [7.462, -3.69, 0, 0, 0.689, 0.724]  # 去第一个房间 有人total_person_count+1
            there_is_a_person = self.A_set_of_processes(req, room1, room=1, biggest_find_person_count=160)
            if there_is_a_person:
                total_person_count += 1
            room2 = [4.438, -6.586, 0, 0, -0.732, 0.6808]  # 去第二个房间 有人total_person_count+1
            there_is_a_person = self.A_set_of_processes(req, room2, room=2, biggest_find_person_count=160)
            if there_is_a_person:
                total_person_count += 1
            room3_point1 = [6.745, -5.21, 0, 0, -0.087, 0.996]  # 去第三个房间第一个点 有人则检测总人数是否达到3 达到则退出程序 没人去第二个点
            should_not_go_another_point = self.A_set_of_processes(req, room3_point1, room=3, biggest_find_person_count=160,  # 60
                                                                  min_person_distance=3.5)
            if should_not_go_another_point:
                total_person_count += 1
                if total_person_count == 3:
                    return None
            else:  # 第三个房间第一个点没人 去第二个点 第二个点有人则检测总人数是否达到3 达到则退出程序
                room3_point2 = [9.427, -6.055, 0, 0, -0.7026, 0.7115]
                there_is_a_person = self.A_set_of_processes(req, room3_point2, room=3, biggest_find_person_count=160)
                if there_is_a_person:
                    total_person_count += 1
                    if total_person_count == 3:
                        return None

            room4_point1 = [4.936, -4.811, 0, 0, 0.9987, 0.0491]  # 去第四个房间第一个点 有人则退出程序 没人去第二个点
            should_not_go_another_point = self.A_set_of_processes(req, room4_point1, room=4, biggest_find_person_count=160,
                                                                  min_person_distance=3)
            if should_not_go_another_point:
                return None
            else:
                room4_point2 = [2.3765, -3.495, 0, 0, 0.7094, 0.7047]  # 去第四个房间第二个点 然后直接退出程序
                self.A_set_of_processes(req, room4_point2, room=4, biggest_find_person_count=160)
                return None

        elif type == 2:
            point = [4.438, -6.586, 0, 0, -0.732, 0.6808]
            # 一套流程是 找人 没找到就旋转 找到人之后靠近 导航到人的正面并距离人一定位置 行为识别 抓取物品
            should_not_go_another_point = self.A_set_of_processes(req, point, room=1, biggest_find_person_count=160, min_person_distance=3.5)
            # if not should_not_go_another_point:
            #     point = [9.427, -6.055, 0, 0, -0.7026, 0.7115]
            #     self.A_set_of_processes(req, point, biggest_find_person_count=160)
            print('over', '!' * 50, round(time.time() - self.start_time))


def doReq(req):
    # 解析提交的数据
    need = req.isget
    resp = poseResponse()
    print(need)
    if need == 1:
        rospy.loginfo("成功")
        Body_detection_path = r'/home/gardenia/zhongji_ws/vision/src/liu/yolov8n-pose.onnx'
        serial_port = '/dev/ttyUSB2'
        BD = BehavioralDetection(Body_detection_path, serial_port)
        BD.start(req, type=1)
        print('over', '!' * 50, round(time.time() - BD.start_time))
        BD.pyrs.pipeline.stop()
        print('退出函数了')
        print(resp)
    return resp


def laser_callback(scan):
    global front_distance
    selected_ranges1 = scan.ranges[0:25]
    selected_ranges2 = scan.ranges[335:360]
    lst = [selected_ranges1[0], selected_ranges2[0]]
    front_distance = min(lst)  # 找到最小距离
    # rospy.loginfo("后方最小距离: %f", front_distance)  # 使用 ROS 日志打印距离
    # distance_queue.put(front_distance)


def laser_listener():
    # 订阅激光雷达的/scan话题
    rospy.Subscriber("/scan", LaserScan, laser_callback)
    rospy.loginfo("激光雷达监听器已启动，等待数据...")
    rospy.spin()  # 保持节点活动


def service_thread():
    server = rospy.Service("test01", pose, doReq)
    print("pull sleeping!")
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node("gain_object_currencydasdas")
    # 创建并启动周期性函数线程
    laser_thread = threading.Thread(target=laser_listener)
    laser_thread.daemon = True  # 设置为守护线程
    # 定义一个队列来存储共享数据
    # distance_queue = Queue()
    front_distance = 0
    laser_thread.start()
    # 启动服务线程
    service_thread()
