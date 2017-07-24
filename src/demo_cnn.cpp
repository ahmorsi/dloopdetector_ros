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

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

// ----------------------------------------------------------------------------

static const char *VOC_FILE = "/resources/surf64_k10L6.voc.gz";
static const char *IMAGE_DIR = "/resources/images";
static const char *POSE_FILE = "/resources/pose.txt";
static const int IMAGE_W = 640; // image size
static const int IMAGE_H = 480;

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
    vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int main()
{
  std::string path = ros::package::getPath("dloopdetector");
  std::string voc_file = path + VOC_FILE;
  std::string image_dir = path + IMAGE_DIR;
  std::string pose_file = path + POSE_FILE;
    // prepares the demo
  demoDetector<Surf64Vocabulary, Surf64LoopDetector, FSurf64::TDescriptor>
    demo(voc_file.c_str(), image_dir.c_str(), pose_file.c_str(), IMAGE_W, IMAGE_H);

  try
  {
    // run the demo with the given functor to extract features
    SurfExtractor extractor;
    demo.run("SURF64", extractor);
  }
  catch(const std::string &ex)
  {
    cout << "Error: " << ex << endl;
  }

  return 0;
}

// ----------------------------------------------------------------------------

void CnnExtractor::operator() (const cv::Mat &im,
  vector<cv::KeyPoint> &keys, vector<vector<float> > &descriptors) const
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


