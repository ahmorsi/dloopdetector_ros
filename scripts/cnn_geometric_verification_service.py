#!/usr/bin/env python

import rospy
from dloopdetector.srv import *
from dloopdetector.msg import *
import os
from cnn_geometric_verification import CNNGeometricMatcher,read_input,read_features_input
import torch
import argparse

class GeometricMatchService:
    def __init__(self,data_folder,geometric_affine_model,image_shape,geometric_tps_model=None,use_extracted_features=False,arch = 'resnet18',
                                    featext_weights = None,service_name='match_two_places'):
        self.service_name = service_name
        self.data_folder=data_folder
        self.image_shape=image_shape
        self.use_extracted_features = use_extracted_features
        self.matcher = CNNGeometricMatcher(use_extracted_features=use_extracted_features, geometric_affine_model=geometric_affine_model,
                                           geometric_tps_model=geometric_tps_model,
                                           arch=arch,featext_weights=featext_weights,
                                      min_mutual_keypoints=3,min_reprojection_error=100)
    def start(self):
        rospy.init_node('cnn_geometric_node')
        s = rospy.Service(self.service_name, MatchTwoPlaces, self.handle_match_two_places)
        print "Ready to match two places."
        rospy.spin()
    def handle_match_two_places(self,req):
        if self.use_extracted_features:
            path_tensorA = os.path.join(self.data_folder,'{}.pt'.format(req.a))
            path_tensorB = os.path.join(self.data_folder, '{}.pt'.format(req.b))
            batch = read_features_input(path_tensorA=path_tensorA,path_tensorB=path_tensorB,image_shape=self.image_shape)
        else:
            path_imgA = os.path.join(self.data_folder,'{}.png'.format(req.a))
            path_imgB = os.path.join(self.data_folder,'{}.png'.format(req.b))
            batch = read_input(path_imgA=path_imgA,path_imgB=path_imgB)

        reprojection_error, matched, num_mutual_keypoints = self.matcher.run(batch)
        result = GeometricSimilarity()
        result.matched = matched
        result.reprojection_error = reprojection_error
        result.num_mutual_matches = num_mutual_keypoints
        print('Match {} & {} - Matched: {} ,Mutual Keypoints: {}, Error: {}'.format(req.a, req.b,matched,num_mutual_keypoints,reprojection_error))
        return MatchTwoPlacesResponse(result)

if __name__ == "__main__":
    #Todo: Add Config File for Service & Remove Data Folder Lists for Image Reading
    parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
    # Paths
    parser.add_argument('--model-aff', type=str,
                        default='/home/develop/Work/Source_Code/cnngeometric_pytorch/trained_models/best_streetview_checkpoint_adam_tps_grid_loss.pth.tar',
                        help='Trained affine model filename')
    parser.add_argument('--model-tps', type=str,
                        default='trained_models/best_streetview_checkpoint_adam_tps_grid_loss.pth.tar',
                        help='Trained TPS model filename')
    parser.add_argument('--features-dir', '-f', type=str,
                        required=True, dest="featuresdir",
                        help='Features Output Folder')

    parser.add_argument('--use-features',action="store_true",help="Use Extracted Features")
    parser.add_argument('--w', type=int,default=640,required=False,help='Image Width')
    parser.add_argument('--h', type=int,default=480, required=False, help='Image Height')

    args = parser.parse_args()
    service = GeometricMatchService(
                                    geometric_affine_model=args.model_aff,
                                    data_folder=args.featuresdir,
                                    use_extracted_features=args.use_features,
                                    image_shape = (args.h,args.w,3))

    service.start()