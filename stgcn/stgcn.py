import os
import torch
import numpy as np

from .Models import TwoStreamSpatialTemporalGraph
from .Utils import normalize_points_with_size, scale_pose

'''
Two-Stream Spatial Temporal Graph Model（双流时空图模型）是一种用于动作识别的深度学习模型。
它结合了时空建模和图神经网络的思想，用于从视频或骨架序列中预测人体动作
'''

'''
该模型的名称中包含了几个关键概念：

Two-Stream（双流）：模型使用两个输入流，分别是空间流（spatial stream）和时间流（temporal stream）。
空间流处理空间信息，通常是从图像中提取的骨架点或关键点，而时间流则处理时间序列信息，例如骨架点的时间演变。
Spatial Temporal（时空）：模型同时考虑了空间和时间信息，以捕捉动作的空间和时间特征。通过综合空间和时间的信息，模型可以更好地理解和预测动作。
Graph Model（图模型）：模型使用图结构来表示人体姿势或骨架关系。每个关键点或骨架节点都被视为图中的一个节点，节点之间的连接表示它们之间的关系。图模型可以有效地捕捉姿势之间的关联和依赖关系。
Two-Stream Spatial Temporal Graph Model通常通过深度学习技术进行训练，包括卷积神经网络（CNN）和图神经网络（GNN）。它可以在视频或骨架序列中预测各种人体动作，例如站立、行走、坐下、躺下等。该模型在动作识别、动作分析和人体行为理解等领域具有广泛的应用。
'''
class STGCN(object):
    """Two-Stream Spatial Temporal Graph Model Loader.
    Args:
        weight_file: (str) Path to trained weights file.
        device: (str) Device to load the model on 'cpu' or 'cuda'.
    """
    def __init__(self,
                 weight_file='./Models/TSSTG/tsstg-model.pth',
                 device='cuda'):
        self.graph_args = {'strategy': 'spatial'}
        self.class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
                            'Stand up', 'Sit down', 'Fall Down']
        self.num_class = len(self.class_names)
        self.device = device

        self.model = TwoStreamSpatialTemporalGraph(self.graph_args, self.num_class).to(self.device)
        self.model.load_state_dict(torch.load(weight_file,  map_location=torch.device(device)))
        self.model.eval()

    def predict(self, pts, image_size):
        """Predict actions from single person skeleton points and score in time sequence.
        Args:
            pts: 形状为(t, v, c)的骨架点和分数的numpy数组 where
                t：输入序列的时间步数. ,
                v：图节点的数量（身体部位）,
                c：通道数（x、y、分数）,
            image_size：图像帧的宽度和高度，以元组形式表示.
       返回值：
            (numpy array)：每个类别动作的概率。
        """
        #  将骨架点的(x, y)坐标归一化为图像尺寸image_size的范围内
        pts[:, :, :2] = normalize_points_with_size(pts[:, :, :2], image_size[0], image_size[1])
        #  对归一化后的坐标进行尺度调整，通过调用scale_pose函数实现。
        pts[:, :, :2] = scale_pose(pts[:, :, :2])
        #  将处理后的骨架点和原始骨架点的中心点坐标进行连接，以获得更全面的姿势信息
        pts = np.concatenate((pts, np.expand_dims((pts[:, 1, :] + pts[:, 2, :]) / 2, 1)), axis=1)

        pts = torch.tensor(pts, dtype=torch.float32)
        # 对pts进行维度转换，通过调用permute方法，将维度从(t, v, c)转换为(c, t, v)。这样做是为了符合模型输入的要求。
        pts = pts.permute(2, 0, 1)[None, :]
        # 使用转换后的pts计算运动信息mot。
        # mot的计算是通过对pts的前一帧和当前帧的坐标进行差分计算得到的。
        # 具体地，取pts的第0维和第1维的前-1个索引（即去掉最后一个时间步）和从第1维的第1个索引开始到最后一个索引，然后计算这两部分的差异
        mot = pts[:, :2, 1:, :] - pts[:, :2, :-1, :]
        mot = mot.to(self.device)
        pts = pts.to(self.device)

        out = self.model((pts, mot))

        return out.detach().cpu().numpy()
