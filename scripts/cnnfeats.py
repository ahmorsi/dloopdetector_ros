#!/usr/bin/env python
import sys
import os
import rospy
import cv2
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, UInt16, Int32, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dloopdetector.msg import *
from keras.applications import vgg16, resnet50
from keras.models import Model,load_model
from keras.layers import Input, Lambda
from keras import backend as K
from sklearn.neighbors import KDTree
import numpy as np
import math
import argparse
import tensorflow as tf

def Vgg16FeatExtModel(input_tensor, weightsPath, output_layer_name):
    vgg16_model = vgg16.VGG16(weights=None, include_top=False, input_tensor=input_tensor)
    vgg16_model.load_weights(weightsPath)
    features_layer = vgg16_model.get_layer(output_layer_name).output
    # output_norm_layer = Lambda(lambda x: K.l2_normalize(x, axis=2))(features_layer)
    model = Model(inputs=vgg16_model.input, outputs=features_layer)
    return model


def ResNetFeatExModel(input_tensor, weightsPath, output_layer_name):
    resnet_model = resnet50.ResNet50(weights=None, include_top=False, input_tensor=input_tensor)
    resnet_model.load_weights(weightsPath)
    features_layer = resnet_model.get_layer(output_layer_name).output
    model = Model(inputs=resnet_model.input, outputs=features_layer)
    return model

def ResNet18FeatExModel(model_path):
    model = load_model(model_path,compile=False)
    return model

class CnnFeaturesPublisher:
    def __init__(self, topic_name, queue_size=10):
        self.pub = rospy.Publisher(topic_name, FeaturesWithKeyPoints, queue_size=queue_size)

    def publish(self, features_keypoints):
        self.pub.publish(features_keypoints)



class CnnFeatureExtractor:
    def __init__(self, arch,weights,output_size=(224, 224),features_dir="features",image_topic_name="/image",features_topic_name="/features", queue_size=10):
        if arch == 'resnet18':
            self.featExtractor = ResNet18FeatExModel(model_path=weights)
        elif arch == 'resnet50':
            self.featExtractor = ResNetFeatExModel(input_tensor=Input((output_size[0],output_size[1],3)),weightsPath=weights,output_layer_name="add_15")
        elif arch == 'vgg16':
            self.featExtractor = Vgg16FeatExtModel(input_tensor=Input((output_size[0], output_size[1], 3)),
                                                   weightsPath=weights, output_layer_name="block5_conv1")

        self.__graph = tf.get_default_graph()
        self.feats_pub = CnnFeaturesPublisher(features_topic_name,queue_size)
        self.output_size = output_size
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
        if image is None:
            return

        resized_img = cv2.resize(src=image,dsize=self.output_size)
        resized_img = np.expand_dims(resized_img,0)

        with self.__graph.as_default():
            output = self.featExtractor.predict(resized_img)

        features = output[0]
        n, m, d = features.shape[:]

        print("Saving Frame {0}".format(self.count))
        cv2.imwrite(os.path.join(self.features_dir,"{0}.png".format(self.count)),image)

        features_msg = Float64MultiArray()
        msg = FeaturesWithKeyPoints()

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
    parser = argparse.ArgumentParser(description='CNN Feature Extraction Keras implementation')
    # Paths
    parser.add_argument('--arch', type=str,
                        default='resnet18',
                        help='Arch of CNN Model (resent18,resnet50,vgg16)')
    parser.add_argument('--weights', type=str,
                        default='/home/develop/Work/Source_Code/CNNs/resnet18/Places302_resnet18_add6.h5',
                        help='Weights of Pretrained Model')

    parser.add_argument('--features-dir','-f', type=str,
                        required=True,dest="featuresdir",
                        help='Images/Features Temporary Output Folder')
    args = parser.parse_args()

    rospy.init_node('FeatExt')
    r = rospy.Rate(5)  # 5hz
    sub_pub_CnnFeatureExtractor = CnnFeatureExtractor(arch=args.arch,
                                                      weights=args.weights,
                                                    features_dir=args.featuresdir)

    sub_pub_CnnFeatureExtractor.start()

    #
    # # model = Vgg16FeatExtModel(input_tensor=Input(shape=(224, 224, 3)),
    # #                           weightsPath="/home/develop/Work/Source_Code/CNNs/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
    # #                           output_layer_name="block3_pool")
    #
    # model = ResNet18FeatExModel(model_path="/home/develop/Work/Source_Code/CNNs/resnet18/Places302_resnet18_add6.h5")
    # imgs_dir = '/home/develop/Work/catkin_ws/src/dloopdetector/resources/images'
    # imgFileNames = sorted([f for f in os.listdir(imgs_dir) if
    #                        os.path.isfile(os.path.join(imgs_dir, f)) and (f.endswith(".jpg") or f.endswith(".png"))])
    # img_width, img_height = 224, 224
    # count = 1
    # # Initiate FAST object with default values
    # fast = cv2.FastFeatureDetector_create(threshold=30)
    #
    # for imgfile in imgFileNames:
    #     gray_img = cv2.imread(os.path.join(imgs_dir, imgfile), 0)
    #     gray_resized_img = cv2.resize(gray_img, (img_width, img_height))
    #     # keypoints = fast.detect(gray_resized_img, None)
    #     #keypoints = fast.detect(gray_img, None)
    #     #if len(keypoints) == 0:
    #     #    continue
    #     #kps = np.zeros((len(keypoints), 2))
    #     #for i in range(len(keypoints)):
    #     #    kps[i] = keypoints[i].pt
    #
    #     net_img = cv2.cvtColor(gray_resized_img, cv2.COLOR_GRAY2BGR)
    #     img_arr = net_img.reshape((1,) + net_img.shape)
    #     output = model.predict(img_arr)
    #     features = output[0]
    #     # keypoints = []
    #     n, m, d = features.shape[:]
    #     feats = np.zeros((n * m, d))  # features.reshape((np.prod(features.shape[:2]),features.shape[2]))
    #     # ratio_width,ratio_height = img_width/n,img_height/m
    #
    #     ratio_width, ratio_height = gray_img.shape[0] // n, gray_img.shape[1] // m
    #     features_msg = Float64MultiArray()
    #     #kdtree = KDTree(kps)
    #     #keypoint_msg = KeyPoint()
    #     msg = FeaturesWithKeyPoints()
    #     #img_keypoints = cv2.drawKeypoints(gray_img, keypoints, color=(255, 0, 0), outImage=None)
    #     #cv2.imshow('img_keypoints', img_keypoints)
    #
    #     radius = math.sqrt(ratio_width ** 2 + ratio_height ** 2) // 2.0
    #     for i in range(n):
    #         for j in range(m):
    #             # ptx, pty = (i * ratio_width) + (ratio_width*0.5), (j * ratio_height) + (ratio_height*0.5)
    #             features_msg.data.extend(features[i, j, :].astype(np.float64))
    #             # ptx, pty = (i * ratio_width), (j * ratio_height)
    #             # ind = kdtree.query_radius(np.array([[ptx, pty]]), r=radius)
    #             # if ind[0].shape[0] > 0:
    #             #     # median_pt = np.median(kps[ind[0]],axis=0)
    #             #     # median_ptx , median_pty = median_pt[0] , median_pt[1]
    #             #     # print((i,j))
    #             #     features_msg.data.extend(features[i, j, :].astype(np.float64))
    #             #     keypointmap_msg = KeyPointMap()
    #             #     # keypoints.append(cv2.KeyPoint(x=ratio_width*i,y=ratio_height*j,_size=32))
    #             #     local_keypoints = [keypoints[idx] for idx in ind[0]]
    #             #     for keypnt in local_keypoints:
    #             #         # keypnt = keypoints[ind[0][0]]
    #             #         keypoint_msg.x, keypoint_msg.y = keypnt.pt  # int(median_ptx)
    #             #         # keypoint_msg.y = keypnt.pt#int(median_ptx)
    #             #         keypoint_msg.size = float(keypnt.size)
    #             #         keypoint_msg.angle = float(keypnt.angle)
    #             #         keypoint_msg.response = float(keypnt.response)
    #             #         keypoint_msg.octave = int(keypnt.octave)
    #             #         keypoint_msg.class_id = int(keypnt.class_id)
    #             #         keypointmap_msg.keypoints.append(keypoint_msg)
    #             #     msg.keypointmaps.append(keypointmap_msg)
    #                 # features_msg.data.extend(features[keypoint_msg.x/ratio_width, keypoint_msg.y/ratio_height, :].astype(np.float64))
    #
    #     # if len(msg.keypointmaps) == 0:
    #     #     continue
    #     features_msg.layout.dim.append(MultiArrayDimension())
    #     features_msg.layout.dim[0].size = n*m#len(msg.keypointmaps)
    #     features_msg.layout.dim[0].stride = len(features_msg.data)  # n*m*d
    #     features_msg.layout.dim[0].label = 'height'
    #     features_msg.layout.dim.append(MultiArrayDimension())
    #     features_msg.layout.dim[1].stride = d
    #     features_msg.layout.dim[1].size = d
    #     features_msg.layout.dim[1].label = 'width'
    #
    #     msg.features = features_msg
    #     feats_pub.publish(msg)
    #     print(imgfile + ' ' + str(count))
    #     count += 1
    #     cv2.imshow("Img",gray_resized_img)
    #     cv2.waitKey(300)
    #     r.sleep()
