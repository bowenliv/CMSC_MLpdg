import math
from collections import deque

import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import models
from resnet_ori import resnet50

class conv_bn_relu(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding,
            groups=1, bias=False, bn=True, relu=True):
        super(conv_bn_relu, self).__init__()
        self.has_bn = bn
        self.has_relu = relu

        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

        if self.has_bn:
            self.bn = nn.BatchNorm2d(output_channel)
        if self.has_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

class ImageEncoder(nn.Module):
    """ 
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            conv_bn_relu(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
        )

    def forward(self, inputs):
        features = self.features(inputs)
        return features

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class LidarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, in_channels=2):
        super().__init__()

        self.features = nn.Sequential(
            conv_bn_relu(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            conv_bn_relu(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
        )

    def forward(self, inputs):
        features = self.features(inputs)

        return features


class Encoder(nn.Module):

    def __init__(self, ):
        super().__init__()
        
        self.image_encoder = ImageEncoder()
        self.lidar_encoder = LidarEncoder(in_channels=2)
        self.share_arch = resnet50()
        
    def forward(self, image_list, lidar_list):


        image_list = [image_input for image_input in image_list]

        bz, _, h, w = lidar_list[0].shape
        img_channel = image_list[0].shape[1]
        lidar_channel = lidar_list[0].shape[1]

        image_tensor = torch.stack(image_list, dim=1).view(bz, img_channel, h, w)
        lidar_tensor = torch.stack(lidar_list, dim=1).view(bz, lidar_channel, h, w)
        image_features = self.image_encoder(image_tensor)
        lidar_features = self.lidar_encoder(lidar_tensor)
        fused_features = torch.cat([image_features, lidar_features], dim=1)
        fused_features = self.share_arch(fused_features)

        return fused_features


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class SAC_AD(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pred_len = config.pred_len

        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        self.encoder = Encoder()

        self.join = nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 64),
                            nn.ReLU(inplace=True),
                        )
        self.decoder = nn.GRUCell(input_size=2, hidden_size=64)
        self.output = nn.Linear(64, 2)
        
    def forward(self, image_list, lidar_list, target_point):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            velocity (tensor): input velocity from speedometer
        '''

        fused_features = self.encoder(image_list, lidar_list)

        z = self.join(fused_features)

        output_wp = list()

        # initial input variable to GRU
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).cuda()

        # autoregressive generation of output waypoints
        for _ in range(self.pred_len):
            # x_in = torch.cat([x, target_point], dim=1)
            x_in = x + target_point
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp

    def control_pid(self, waypoints, velocity):
        ''' 
        Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): predicted waypoints
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()

        # flip y is (forward is negative in our waypoints)
        waypoints[:,1] *= -1
        speed = velocity[0].data.cpu().numpy()

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
        brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        if(speed < 0.01):
            angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.config.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'delta': float(delta.astype(np.float64)),
        }

        return steer, throttle, brake, metadata