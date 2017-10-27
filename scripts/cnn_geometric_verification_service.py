#!/usr/bin/env python

import rospy
from dloopdetector.srv import *
from dloopdetector.msg import *
import os
from cnn_geometric_verification import CNNGeometricMatcher,read_input
import torch

class GeometricMatchService:
    def __init__(self,model_path,data_folder,service_name='cnn_geometric_verification_service'):
        rospy.init_node(service_name)
        self.imgFileNames = sorted([f for f in os.listdir(data_folder) if
                               os.path.isfile(os.path.join(data_folder, f)) and (
                                   f.endswith(".jpg") or f.endswith(".png"))])
        self.matcher = CNNGeometricMatcher(use_cuda=False, geometric_model='affine', model_path=model_path,
                                      min_mutual_keypoints=4)
    def start(self):
        s = rospy.Service('match_two_places', MatchTwoPlaces, self.handle_match_two_places)
        print "Ready to match two places."
        rospy.spin()
    def handle_match_two_places(self,req):

        path_imgA = self.imgFileNames[req.a]
        path_imgB = self.imgFileNames[req.b]
        batch = read_input(path_imgA=path_imgA,path_imgB=path_imgB)
        reprojection_error, matched, num_mutual_keypoints = self.matcher.run(batch)
        result = GeometricSimilarity()
        result.matched = matched
        result.reprojection_error = reprojection_error
        result.num_mutual_matches = num_mutual_keypoints
        return MatchTwoPlacesResponse(result)

if __name__ == "__main__":
    service = GeometricMatchService(model_path="/home/develop/Work/Source_Code/cnngeometric_pytorch/trained_models/best_streetview_checkpoint_adam_affine_grid_loss.pth.tar",
                                    data_folder="/home/develop/Work/catkin_ws/src/dloopdetector/resources/images")

    service.start()