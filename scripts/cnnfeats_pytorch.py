#!/usr/bin/env python
import sys
import os
import rospy
import cv2
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, UInt16, Int32, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dloopdetector.msg import *
import numpy as np
import math
from model.cnn_geometric_model import CNNGeometric,CNNGeometricRegression,FeatureExtraction
import argparse
import torch
from torch.autograd import Variable
from geotnf.transformation import GeometricTnf

class CnnFeaturesPublisher:
    def __init__(self, topic_name, queue_size=10):
        self.pub = rospy.Publisher(topic_name, FeaturesWithKeyPoints, queue_size=queue_size)

    def publish(self, features_keypoints):
        self.pub.publish(features_keypoints)

class CnnFeatureExtractor:
    def __init__(self, arch,weights,output_size=(240, 240),features_dir="features",image_topic_name="/image",features_topic_name="/features", queue_size=10):
        self.featExtractor = FeatureExtraction(arch=arch, weights=weights,use_cuda=torch.cuda.is_available())
        self.feats_pub = CnnFeaturesPublisher(features_topic_name,queue_size)
        # subscribed Topic/
        self.subscriber = rospy.Subscriber(image_topic_name,
                                           Image, self.callback, queue_size=10)
        out_h, out_w = output_size
        self.resizeTnf = GeometricTnf(out_h=out_h, out_w=out_w, use_cuda=False)
        self.count = 0
        self.features_dir = features_dir
        if not os.path.exists(self.features_dir):
            print("Created Output Features Dir: " + self.features_dir)
            os.makedirs(self.features_dir)
        self.image_topic_name = image_topic_name
        self.bridge = CvBridge()
    def start(self):
        self.subscriber = rospy.Subscriber(self.image_topic_name,
                                           Image, self.callback, queue_size=100)
        print('Ready to Extract CNN Features')
        rospy.spin()
    def callback(self,ros_data):

        image = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
        print(image.shape)
        image = np.expand_dims(np.transpose(image,(2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image, requires_grad=False)

        image = self.resizeTnf(image_var)

        features_tensor = self.featExtractor(image)
        print("Save Features of Frame {0}".format(self.count))
        torch.save(features_tensor.data,os.path.join(self.features_dir,"{0}.pt".format(self.count)))

        features_np = features_tensor.data.numpy()[0]
        features = np.transpose(features_np,(1, 2, 0))

        features_msg = Float64MultiArray()
        msg = FeaturesWithKeyPoints()

        n, m, d = features.shape[:]
        for i in range(n):
            for j in range(m):
                features_msg.data.extend(features[i, j, :].astype(np.float64))

        features_msg.layout.dim.append(MultiArrayDimension())
        features_msg.layout.dim[0].size = n*m#len(msg.keypointmaps)
        features_msg.layout.dim[0].stride = len(features_msg.data)  # n*m*d
        features_msg.layout.dim[0].label = 'height'
        features_msg.layout.dim.append(MultiArrayDimension())
        features_msg.layout.dim[1].stride = d
        features_msg.layout.dim[1].size = d
        features_msg.layout.dim[1].label = 'width'

        msg.features = features_msg
        self.feats_pub.publish(msg)
        print("Published Features of Frame {0}".format(self.count))
        self.count += 1

if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='CNN Feature Extraction PyTorch implementation')
    # Paths
    parser.add_argument('--arch', type=str,
                        default='resnet18',
                        help='Arch of CNN Model (resent18,resnet50,vgg16)')
    parser.add_argument('--weights', type=str,
                        default='/home/develop/Work/Source_Code/CNNs/resnet18/Places302_pytorch_model_best.pth.tar',
                        help='Weights of Pretrained Model')

    parser.add_argument('--features-dir','-f', type=str,
                        required=True,dest="featuresdir",
                        help='Features Output Folder')
    args = parser.parse_args()

    rospy.init_node('FeatExt')
    r = rospy.Rate(5)  # 5hz

    sub_pub_CnnFeatureExtractor = CnnFeatureExtractor(arch=args.arch,
                                                      weights=args.weights,
                                                    features_dir=args.featuresdir)

    sub_pub_CnnFeatureExtractor.start()
