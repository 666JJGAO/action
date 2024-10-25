import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from stgcn.stgcn import STGCN
from PIL import Image, ImageDraw, ImageFont


def cv2_add_chinese_text(word_path, img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式 "./fonts/MSYH.ttc"
    fontStyle = ImageFont.truetype(
        word_path, textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

class my_pose_point:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles  # 定义了一些预定义的绘制样式
        self.mp_pose = mp.solutions.pose  # 进行姿势估计的主要模块。通过使用此模块，您可以加载预训练的姿势估计模型，并将图像或视频中的人体姿势进行估计
        self.KEY_JOINTS = [
    self.mp_pose.PoseLandmark.NOSE,  # 鼻子 1
    self.mp_pose.PoseLandmark.LEFT_SHOULDER,  # 左肩膀 2
    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,  # 右肩膀 3
    self.mp_pose.PoseLandmark.LEFT_ELBOW,  # 左肘 4
    self.mp_pose.PoseLandmark.RIGHT_ELBOW,  # 右肘 5
    self.mp_pose.PoseLandmark.LEFT_WRIST,  # 左手腕 6
    self.mp_pose.PoseLandmark.RIGHT_WRIST,  # 右手腕 7
    self.mp_pose.PoseLandmark.LEFT_HIP,  # 左髋部 8
    self.mp_pose.PoseLandmark.RIGHT_HIP,  # 右髋部 9
    self.mp_pose.PoseLandmark.LEFT_KNEE,  # 左膝盖 10
    self.mp_pose.PoseLandmark.RIGHT_KNEE,  # 右髋部 11
    self.mp_pose.PoseLandmark.LEFT_ANKLE,  # 左脚踝 12
    self.mp_pose.PoseLandmark.RIGHT_ANKLE  # 右脚踝 13
]

        '''
        POSE_CONNECTIONS的列表，其中包含一些用于连接姿势关节点的索引对。这些索引对用于绘制姿势关节点之间的连线，以创建人体姿势的可视化效果。
        '''
        self.POSE_CONNECTIONS = [(6, 4), (4, 2), (2, 13), (13, 1), (5, 3), (3, 1), (12, 10),
                    (10, 8), (8, 2), (11, 9), (9, 7), (7, 1), (13, 0)] # 一共13个


        self.POINT_COLORS = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, 左手腕, 右手腕
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # 左髋部, RHip, 左髋部, Rknee, 左脚踝, RAnkle, Neck

        self.LINE_COLORS = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222),
               (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255),
               (255, 156, 127), (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    def draw_skeleton(self,frame, pts):
        l_pair = self.POSE_CONNECTIONS
        p_color = self.POINT_COLORS
        line_color = self.LINE_COLORS

        part_line = {}
        pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
        for n in range(pts.shape[0]):
            if pts[n, 2] <= 0.05:
                continue
            cor_x, cor_y = int(pts[n, 0]), int(pts[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(frame, (cor_x, cor_y), 3, p_color[n], -1)
            # cv2.putText(frame, str(n), (cor_x+10, cor_y+10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(frame, start_xy, end_xy, line_color[i], int(1 * (pts[start_p, 2] + pts[end_p, 2]) + 3))
        return frame




class FallDetection:
    def __init__(self,model_path):
        self.action_model = STGCN(weight_file=model_path, device='cuda')


    def action_re(self, joints_list, image_w, image_h):
        # 识别动作
        action = ''
        # 30帧数据预测动作类型
        # 最大帧数
        ACTION_MODEL_MAX_FRAMES = 30
        if len(joints_list) == ACTION_MODEL_MAX_FRAMES:
            pts = np.array(joints_list, dtype=np.float32)
            out = self.action_model.predict(pts, (image_w, image_h))
            action_name = self.action_model.class_names[out[0].argmax()]
            action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
            print(action)
            if action_name == 'Fall Down':
                action = '摔倒'
            elif action_name == 'Walking':
                action = 'walking'
            elif action_name == 'Sitting':
                action = 'sitting'
            elif action_name == 'Standing':
                action = 'standing'
            elif action_name == 'Lying Down':
                action = 'lay'

            else:
                action = ''

            return action_name, action




if __name__ == '__main__':
    word="./fonts/MSYH.ttc"
    FallDetection('./weights/tsstg-model.pth').detect()
