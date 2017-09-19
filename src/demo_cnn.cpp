/**
 * File: demo_surf.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <string>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines Surf64Vocabulary
#include "DLoopDetector.h" // defines Surf64LoopDetector
#include <DUtilsCV/DUtilsCV.h> // defines macros CVXX

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// Demo
#include "demoDetector.h"

//msg
#include "dloopdetector/FeaturesWithKeyPoints.h"
#include "dloopdetector/KeyPoint.h"
#include "ros/ros.h"
using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

// ----------------------------------------------------------------------------

static const char *VOC_FILE = "/resources/library_resnet_voc_K10L6.txt";
static const char *IMAGE_DIR = "/resources/images";
static const char *POSE_FILE = "/resources/pose.txt";
static const int IMAGE_W = 224; // image size
static const int IMAGE_H = 224;

demoDetector<CnnVocabulary, CnnLoopDetector, FCNN::TDescriptor> *demo;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// This functor extracts SURF64 descriptors in the required format
class CnnExtractor: public FeatureExtractor<FCNN::TDescriptor>
{
public:
  /**
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im,
    vector<cv::KeyPoint> &keys,vector<DLoopDetector::KeyPointMap> &keymaps, vector<vector<float> > &descriptors) const;


};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
int cnt = 0;
void extractor_callback(const dloopdetector::FeaturesWithKeyPoints::ConstPtr& msg);

int main(int argc, char **argv)
{
  ros::init(argc, argv, "cnn_dloopdetector");
  ros::NodeHandle n;

  std::string path = ros::package::getPath("dloopdetector");
  std::string voc_file = path + VOC_FILE;
  std::string image_dir = path + IMAGE_DIR;
  std::string pose_file = path + POSE_FILE;
    // prepares the demo
    demo = new demoDetector<CnnVocabulary, CnnLoopDetector, FCNN::TDescriptor>(voc_file.c_str(), image_dir.c_str(), pose_file.c_str(), IMAGE_W, IMAGE_H);
    demo->init("CNN");
  try
  {
    // run the demo with the given functor to extract features
    //CnnExtractor extractor;
    //demo.run("CNN", extractor);
    ros::Subscriber sub = n.subscribe("/features", 1000, extractor_callback);
    ros::spin();
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }

  return 0;
}

// ----------------------------------------------------------------------------

void CnnExtractor::operator() (const cv::Mat &im,
  vector<cv::KeyPoint> &keys,vector<DLoopDetector::KeyPointMap> &keymaps, vector<vector<float> > &descriptors) const
{
  // extract surfs with opencv
  static cv::Ptr<cv::xfeatures2d::SURF> surf_detector =
    cv::xfeatures2d::SURF::create(400);

  surf_detector->setExtended(false);

  keys.clear(); // opencv 2.4 does not clear the vector
  vector<float> plain;
  surf_detector->detectAndCompute(im, cv::Mat(), keys, plain);

  // change descriptor format
  const int L = surf_detector->descriptorSize();
  descriptors.resize(plain.size() / L);

  unsigned int j = 0;
  for(unsigned int i = 0; i < plain.size(); i += L, ++j)
  {
    descriptors[j].resize(L);
    std::copy(plain.begin() + i, plain.begin() + i + L, descriptors[j].begin());
  }
}

// ----------------------------------------------------------------------------
void extractor_callback(const dloopdetector::FeaturesWithKeyPoints::ConstPtr &msg)
{
    vector<cv::KeyPoint> keys;
    vector<DLoopDetector::KeyPointMap> keymaps;
    for(std::vector<dloopdetector::KeyPoint>::const_iterator kp = msg -> keypoints.begin();
        kp != msg-> keypoints.end(); ++ kp)
    {
        keys.push_back(cv::KeyPoint(kp->x,kp->y,kp->size));
    }
    int rows = msg -> features.layout.dim[0].size;
    int cols = msg -> features.layout.dim[1].size;
    const std::vector<double> data_1d = msg->features.data;
    std::vector<FCNN::TDescriptor> descriptors;
    descriptors.resize(rows);
    for(int i = 0;i<rows;++i) {
        descriptors[i].resize(cols);
        int offset = i*cols;
        std::copy(data_1d.begin()+offset,data_1d.begin() + offset + cols,descriptors[i].begin());
    }
    demo->runOnImage(keys,keymaps,descriptors);
    cout<<++ cnt << "-> "<<keys.size()<<' '<< descriptors.size()<<'-'<< descriptors[0].size()<<std::endl;
}

