#!/usr/bin/env python
import sys
import rospy
import cv2
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from dloopdetector.srv import *
from keras.applications import vgg16
from keras.models import Model

def Vgg16FeatExtModel(input_tensor,weightsPath,output_layer_name):
    vgg16_model = vgg16.VGG16(weights=None,include_top=False,input_tensor=input_tensor)
    vgg16_model.load_weights(weightsPath)

    model = Model(inputs=vgg16_model.input,outputs=vgg16_model.get_layer(name=output_layer_name).output)
    return model

if __name__ == '__main__':
    #Create Vgg16 Feature Extractor
    #Init Service
    pass