#!/usr/bin/env python
import sys
import os
import rospy
import cv2
from std_msgs.msg import Float64MultiArray,MultiArrayDimension,UInt16,Int32,Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dloopdetector.msg import *
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Input
import numpy as np
import cv2
def Vgg16FeatExtModel(input_tensor,weightsPath,output_layer_name):
    vgg16_model = vgg16.VGG16(weights=None,include_top=False,input_tensor=input_tensor)
    vgg16_model.load_weights(weightsPath)

    model = Model(inputs=vgg16_model.input,outputs=vgg16_model.get_layer(name=output_layer_name).output)
    return model

class CnnFeaturesPublisher:
    def __init__(self,topic_name,queue_size = 10):
        self.pub = rospy.Publisher(topic_name,FeaturesWithKeyPoints,queue_size=queue_size)
    def publish(self,features_keypoints):
        self.pub.publish(features_keypoints)

if __name__ == '__main__':
    feats_pub = CnnFeaturesPublisher(topic_name="/features")
    rospy.init_node('FeatExt')
    r = rospy.Rate(20)  # 20hz
    model = Vgg16FeatExtModel(input_tensor = Input(shape=(224,224,3)),
                              weightsPath="/home/develop/Work/Source_Code/CNNs/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
                              output_layer_name='block5_conv1')
    imgs_dir = '/home/develop/Work/catkin_ws/src/dloopdetector/resources/images'
    imgFileNames = sorted([f for f in os.listdir(imgs_dir) if
                    os.path.isfile(os.path.join(imgs_dir, f)) and (f.endswith(".jpg") or f.endswith(".png"))])
    img_width,img_height = 224,224
    count = 1
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    for imgfile in imgFileNames:
        gray_img = cv2.imread(os.path.join(imgs_dir,imgfile),0)
        gray_resized_img = cv2.resize(gray_img,(img_width,img_height))
        #keypoints = fast.detect(gray_resized_img, None)
        net_img = cv2.cvtColor(gray_resized_img,cv2.COLOR_GRAY2BGR)
        img_arr = net_img.reshape((1,)+net_img.shape)
        output = model.predict(img_arr)
        features = output[0]
        keypoints = []
        n, m , d = features.shape[:]
        feats = np.zeros((n*m,d))#features.reshape((np.prod(features.shape[:2]),features.shape[2]))
        ratio_width,ratio_height = img_width/n,img_height/m

        features_msg = Float64MultiArray()
        for i in range(n):
            for j in range(m):
                #feats[i * m + j] = features[i, j, :]
                features_msg.data.extend(features[i, j, :].astype(np.float64))
                keypoints.append(cv2.KeyPoint(x=ratio_width*i,y=ratio_height*j,_size=32))

        keypoint_msg = KeyPoint()
        msg = FeaturesWithKeyPoints()

        for keypnt in keypoints:
            keypoint_msg.x = int(keypnt.pt[0])
            keypoint_msg.y = int(keypnt.pt[1])
            keypoint_msg.size = float(keypnt.size)
            keypoint_msg.angle = float(keypnt.angle)
            keypoint_msg.response = float(keypnt.response)
            keypoint_msg.octave = int(keypnt.octave)
            keypoint_msg.class_id = int(keypnt.class_id)

            msg.keypoints.append(keypoint_msg)
            #features_msg.data.extend(features[keypoint_msg.x/ratio_width, keypoint_msg.y/ratio_height, :].astype(np.float64))

        features_msg.layout.dim.append(MultiArrayDimension())
        features_msg.layout.dim[0].size = len(keypoints)
        features_msg.layout.dim[0].stride = len(features_msg.data)#n*m*d
        features_msg.layout.dim[0].label = 'height'
        features_msg.layout.dim.append(MultiArrayDimension())
        features_msg.layout.dim[1].stride = d
        features_msg.layout.dim[1].size = d
        features_msg.layout.dim[1].label = 'width'

        msg.features = features_msg
        feats_pub.publish(msg)
        print(imgfile + ' ' + str(count))
        count +=1
        r.sleep()