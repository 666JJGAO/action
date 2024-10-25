import cv2
from cv2 import getTickCount, getTickFrequency
import numpy as np
import onnxruntime as ort


class Keypoint:
    def __init__(self, model_path, draw=False, show=False, fps=False, need_bone=False):
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,  # 可以选择GPU设备ID，如果你有多个GPU
            }),
            'CPUExecutionProvider',  # 也可以设置CPU作为备选
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.label_name = self.session.get_outputs()[0].name
        # 定义一个调色板数组，其中每个元素是一个包含RGB值的列表，用于表示不同的颜色
        self.palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                            [230, 230, 0], [255, 153, 255], [153, 204, 255],
                            [255, 102, 255], [255, 51, 255], [102, 178, 255],
                            [51, 153, 255], [255, 153, 153], [255, 102, 102],
                            [255, 51, 51], [153, 255, 153], [102, 255, 102],
                            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                            [255, 255, 255]])
        # 定义人体17个关键点的连接顺序，每个子列表包含两个数字，代表要连接的关键点的索引, 1鼻子 2左眼 3右眼 4左耳 5右耳 6左肩 7右肩 8左肘 9右肘 10左手腕 11右手腕 12左髋 13右髋 14左膝 15右膝 16左踝 17右踝
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        # 通过索引从调色板中选择颜色，用于绘制人体骨架的线条，每个索引对应一种颜色
        self.pose_limb_color = self.palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        # 通过索引从调色板中选择颜色，用于绘制人体的关键点，每个索引对应一种颜色
        self.pose_kpt_color = self.palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.draw = draw
        self.show = show
        self.fps = fps
        self.need_bone = need_bone
        if self.show:
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            if self.draw:
                cv2.namedWindow('bone_frame', cv2.WINDOW_NORMAL)

    def resize_image(self, image, size=224):
        height, width = image.shape[:2]

        # 计算缩放比例
        scale = min(size / float(height), size / float(width))

        # 缩放图像
        resized_image = cv2.resize(image, None, fx=scale, fy=scale)

        # 创建黑色背景画布
        canvas = np.zeros((size, size, 3), dtype=np.uint8)

        # 计算填充后的位置
        y_offset = (size - resized_image.shape[0]) // 2
        x_offset = (size - resized_image.shape[1]) // 2

        # 将缩放后的图像放置在中心位置
        canvas[y_offset:y_offset + resized_image.shape[0], x_offset:x_offset + resized_image.shape[1]] = resized_image

        return canvas

    def putfps(self, loop_start, frame):
        loop_time = getTickCount() - loop_start
        total_time = loop_time / (getTickFrequency())
        FPS = int(1 / total_time)
        # 在图像左上角添加FPS文本
        fps_text = f"FPS: {FPS:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 0, 255)  # 红色
        text_position = (10, 30)  # 左上角位置

        cv2.putText(frame, fps_text, text_position, font, font_scale, text_color, font_thickness)
        return frame

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), scaleup=True):
        '''  调整图像大小和两边灰条填充  '''
        shape = im.shape[:2]
        if isinstance(new_shape, int):  # isinstance用来看某个对象是否为目标类型
            new_shape = (new_shape, new_shape)
        # 缩放比例 (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # 只进行下采样 因为上采样会让图片模糊
        if not scaleup:
            r = min(r, 1.0)
        # 计算pad长宽
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 保证缩放后图像比例不变
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        # 在较小边的两侧进行pad, 而不是在一侧pad
        dw /= 2
        dh /= 2
        # 将原图resize到new_unpad（长边相同，比例相同的新图）
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        # 计算上下两侧的padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        # 计算左右两侧的padding
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        # 添加灰条
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im

    def pre_process(self, img):
        # 归一化：将图像数据从0~255缩放到0~1之间，这一步是为了让模型更容易学习。
        img = img / 255.
        # 调整通道顺序：将图像从高度x宽度x通道数（H, W, C）调整为通道数x高度x宽度（C, H, W）的形式。
        # 这样做是因为许多深度学习框架要求输入的通道数在前。
        img = np.transpose(img, (2, 0, 1))
        # 增加一个维度：在0轴（即最前面）增加一个维度，将图像的形状从（C, H, W）变为（1, C, H, W）。
        # 这一步是为了满足深度学习模型输入时需要的批量大小（batch size）的维度，即使这里的批量大小为1。
        data = np.expand_dims(img, axis=0)
        return data

    def xywh2xyxy(self, x):
        ''' 中心坐标、w、h ------>>> 左上点，右下点 '''
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def nms(seelf, dets, iou_thresh):
        # dets: N * M, N是bbox的个数，M的前4位是对应的 左上点，右下点
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 求每个bbox的面积
        order = scores.argsort()[::-1]  # 对分数进行倒排序
        keep = []  # 用来保存最后留下来的bboxx下标
        while order.size > 0:
            i = order[0]  # 无条件保留每次迭代中置信度最高的bbox
            keep.append(i)
            # 计算置信度最高的bbox和其他剩下bbox之间的交叉区域
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算置信度高的bbox和其他剩下bbox之间交叉区域的面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 求交叉区域的面积占两者（置信度高的bbox和其他bbox）面积和的必烈
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 保留ovr小于thresh的bbox，进入下一次迭代。
            inds = np.where(ovr <= iou_thresh)[0]
            # 因为ovr中的索引不包括order[0]所以要向后移动一位
            order = order[inds + 1]
        output = []
        for i in keep:
            output.append(dets[i].tolist())
        return np.array(output)

    def xyxy2xywh(self, a):
        ''' 左上点 右下点 ------>>> 左上点 宽 高 '''
        b = np.copy(a)
        b[:, 2] = a[:, 2] - a[:, 0]  # w
        b[:, 3] = a[:, 3] - a[:, 1]  # h
        return b

    def clip_boxes(self, boxes, shape):
        # 进行一个边界截断，以免溢出
        # 并且将检测框的坐标（左上角x，左上角y，宽度，高度）--->>>（左上角x，左上角y，右下角x，右下角y）
        top_left_x = boxes[:, 0].clip(0, shape[1])
        top_left_y = boxes[:, 1].clip(0, shape[0])
        bottom_right_x = (boxes[:, 0] + boxes[:, 2]).clip(0, shape[1])
        bottom_right_y = (boxes[:, 1] + boxes[:, 3]).clip(0, shape[0])
        boxes[:, 0] = top_left_x  # 左上
        boxes[:, 1] = top_left_y
        boxes[:, 2] = bottom_right_x  # 右下
        boxes[:, 3] = bottom_right_y

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        '''   将预测的坐标信息转换回原图尺度
        :param img1_shape: 缩放后的图像尺度
        :param boxes:  预测的box信息
        :param img0_shape: 原始图像尺度
        '''

        # 将检测框(x y w h)从img1_shape(预测图) 缩放到 img0_shape(原图)
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        boxes[:, 0] -= pad[0]
        boxes[:, 1] -= pad[1]
        boxes[:, :4] /= gain  # 检测框坐标点还原到原图上
        num_kpts = boxes.shape[1] // 3  # 56 // 3 = 18
        for kid in range(2, num_kpts + 1):
            boxes[:, kid * 3 - 1] = (boxes[:, kid * 3 - 1] - pad[0]) / gain
            boxes[:, kid * 3] = (boxes[:, kid * 3] - pad[1]) / gain
        # boxes[:, 5:] /= gain  # 关键点坐标还原到原图上
        self.clip_boxes(boxes, img0_shape)
        return boxes

    def calculate_center(self, points):
        valid_points = [point for point in points if all(point)]
        if not valid_points:
            return None, None

        avg_x = sum(point[0] for point in valid_points) // len(valid_points)
        avg_y = sum(point[1] for point in valid_points) // len(valid_points)

        return avg_x, avg_y

    def plot_skeleton_kpts(self, im, bone_frame, kpts, steps=3):
        num_kpts = len(kpts) // steps  # 51 / 3 =17
        # 画点
        for kid in range(num_kpts):
            r, g, b = self.pose_kpt_color[kid]
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            conf = kpts[steps * kid + 2]
            if conf > 0.5:  # 关键点的置信度必须大于 0.5
                cv2.circle(bone_frame, (int(x_coord), int(y_coord)), 10, (int(r), int(g), int(b)), -1)
                # cv2.putText(bone_frame, str(kid), (int(x_coord), int(y_coord)), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
                cv2.circle(im, (int(x_coord), int(y_coord)), 3, (int(r), int(g), int(b)), -1)

        # 画骨架
        for sk_id, sk in enumerate(self.skeleton):
            r, g, b = self.pose_limb_color[sk_id]
            pos1 = (int(kpts[(sk[0] - 1) * steps]), int(kpts[(sk[0] - 1) * steps + 1]))
            pos2 = (int(kpts[(sk[1] - 1) * steps]), int(kpts[(sk[1] - 1) * steps + 1]))
            conf1 = kpts[(sk[0] - 1) * steps + 2]
            conf2 = kpts[(sk[1] - 1) * steps + 2]
            if conf1 > 0.5 and conf2 > 0.5:  # 对于肢体，相连的两个关键点置信度 必须同时大于 0.5
                cv2.line(bone_frame, pos1, pos2, (int(r), int(g), int(b)), thickness=20)
                # cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=20)
        return bone_frame

    def inference(self, image):
        loop_start = getTickCount()
        if self.draw or self.need_bone:
            width, height = image.shape[:2]
            bone_frame = np.zeros((width, height, 3), dtype=np.uint8)
        img = self.letterbox(image)  # 将图像大小转化为640*640，长宽不一则填充
        data = self.pre_process(img)  # 归一化
        # 预测输出float32[1, 56, 8400]
        pred = self.session.run([self.label_name], {self.input_name: data.astype(np.float32)})[0]
        # [1, 56, 8400]
        pred = pred[0]
        # [56, 8400]
        pred = np.transpose(pred, (1, 0))
        # 置信度阈值过滤
        conf = 0.7
        pred = pred[pred[:, 4] > conf]
        # print(111)
        if len(pred) == 0:
            # print("没有检测到任何关键点")
            if self.need_bone:
                return image, None, None, bone_frame
            else:
                return image, None, None
        else:
            people_count = []
            shoulder_head_counts = []
            # 中心宽高转左上点，右下点
            bboxs = self.xywh2xyxy(pred)
            # NMS处理(非极大值抑制)
            bboxs = self.nms(bboxs, iou_thresh=0.6)
            # 坐标从左上点，右下点 到 左上点，宽，高.
            bboxs = np.array(bboxs)
            bboxs = self.xyxy2xywh(bboxs)
            # print(bboxs[:, :4])
            # 坐标点还原到原图, 这个bboxs就是关键点的坐标了, 只不过是用科学计数法表示, 而且有很多位小数, 需要取整
            bboxs = self.scale_boxes(img.shape, bboxs, image.shape)
            # print(bboxs)
            # 画框 画点 画骨架
            for box in bboxs:
                # 依次为 检测框（左上点，右下点）、置信度、17个关键点
                det_bbox, det_scores, kpts = box[0:4], box[4], box[5:]
                people_bbox1, people_bbox2, people_bbox3, people_bbox4 = int(det_bbox[0]), int(det_bbox[1]), int(
                    det_bbox[2]), int(det_bbox[3])
                # central = [(people_bbox1 + people_bbox3) // 2, (people_bbox2 + people_bbox4) // 2]
                ([nose_x, nose_y], [eye1_x, eye1_y], [eye2_x, eye2_y], [ear1_x, ear1_y], [ear2_x, ear2_y],
                 [shoulder1_x, shoulder1_y], [shoulder2_x, shoulder2_y]) = (
                    [int(kpts[0]), int(kpts[1])] if kpts[2] > 0.5 else [None, None],
                    [int(kpts[3]), int(kpts[4])] if kpts[5] > 0.5 else [None, None],
                    [int(kpts[6]), int(kpts[7])] if kpts[8] > 0.5 else [None, None],
                    [int(kpts[9]), int(kpts[10])] if kpts[11] > 0.5 else [None, None],
                    [int(kpts[12]), int(kpts[13])] if kpts[14] > 0.5 else [None, None],
                    [int(kpts[15]), int(kpts[16])] if kpts[17] > 0.5 else [None, None],
                    [int(kpts[18]), int(kpts[19])] if kpts[20] > 0.5 else [None, None]
                )

                shoulder_head_count = [[nose_x, nose_y], [eye1_x, eye1_y], [eye2_x, eye2_y], [ear1_x, ear1_y],
                                       [ear2_x, ear2_y], [shoulder1_x, shoulder1_y], [shoulder2_x, shoulder2_y]]
                shoulder_head_counts.append(shoulder_head_count)
                avg_x, avg_y = self.calculate_center(shoulder_head_count[:5])
                people_count.append([[people_bbox1, people_bbox2, people_bbox3, people_bbox4], [avg_x, avg_y]])
                label = "Person {:.2f}".format(det_scores)
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # 画框
                cv2.rectangle(image, (people_bbox1, people_bbox2), (people_bbox3, people_bbox4),
                              (0, 0, 255), 2)
                label_x = int(det_bbox[0])
                label_y = int(det_bbox[1]) - 10 if int(det_bbox[1]) - 10 > label_height else int(det_bbox[1]) + 10
                # 人体检测置信度
                if int(det_bbox[1]) < 30:
                    pass
                    cv2.rectangle(image, (label_x, label_y - label_height),
                                  (label_x + label_width + 3, label_y + label_height), (0, 0, 255), -1)
                    cv2.putText(image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    pass
                    cv2.rectangle(image, (label_x, label_y - label_height),
                                  (label_x + label_width + 3, label_y + label_height), (0, 0, 255), -1)
                    cv2.putText(image, label, (int(det_bbox[0]) + 5, int(det_bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 255, 255), 1)
                if self.draw or self.need_bone:
                    # 画点 连线
                    bone_frame = self.plot_skeleton_kpts(image, bone_frame, kpts)
                if self.fps:
                    self.putfps(loop_start, image)
                if self.show:
                    if self.draw:
                        cv2.imshow('bone_frame', bone_frame)
                    cv2.imshow("frame", image)
        if self.need_bone:
            return image, people_count, shoulder_head_counts, bone_frame
        else:
            return image, people_count, shoulder_head_counts
