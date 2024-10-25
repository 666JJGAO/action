import pyrealsense2 as rs
import cv2
import numpy as np


class pyrealsense:
    def __init__(self) -> None:
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming
        self.pipeline.start(self.config)
        # 对齐深度图和彩色图
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        # self.pipeline = pipeline
        # self.align = align

    def get_img(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        self.aligned_depth_frame = aligned_frames.get_depth_frame()
        self.color_frame = aligned_frames.get_color_frame()
        self.depth_intrin = self.aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        if not self.aligned_depth_frame or not self.color_frame:
            print("camera error! ")
            return

        depth_image = np.asanyarray(self.aligned_depth_frame.get_data())
        color_image = np.asanyarray(self.color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        return color_image, depth_colormap

    def get_distance(self, center_point):
        x = center_point[0]
        y = center_point[1]
        try:
            dis = self.aligned_depth_frame.get_distance(x, y)
            return dis
        except:
            print("gain distance failed")
            return None

    def get_3d_camera_coordinate(self, depth_pixel):
        x = depth_pixel[0]
        y = depth_pixel[1]
        dis = self.aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
        # print ('depth: ',dis)       # 深度单位是m
        camera_coordinate = rs.rs2_deproject_pixel_to_point(self.depth_intrin, depth_pixel, dis)
        # print ('camera_coordinate: ',camera_coordinate)
        return dis, camera_coordinate
