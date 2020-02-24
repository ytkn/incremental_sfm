#include <fstream>
#include "opencv2/xfeatures2d.hpp"

using std::ifstream;
using namespace cv;
using namespace cv::xfeatures2d;

struct Setting {
  String rootDir;
  bool showMatches;
  int minHessian;
  int startFrame;
  int displayPcdCycle;
  float matchingRatioThresh;
  float errorRatio;
  float reprojectionErrorThresh;
  Ptr<Feature2D> detector;
};

Setting initSetting(String filename) {
  Setting setting;
  cv::FileStorage fs(filename, cv::FileStorage::READ);
  fs["rootDir"] >> setting.rootDir;
  fs["minHessian"] >> setting.minHessian;
  fs["startFrame"] >> setting.startFrame;
  fs["matchingRatioThresh"] >> setting.matchingRatioThresh;
  fs["errorRatio"] >> setting.errorRatio;
  fs["reprojectionErrorThresh"] >> setting.reprojectionErrorThresh;
  fs["showMatches"] >> setting.showMatches;
  fs["displayPcdCycle"] >> setting.displayPcdCycle;
  // setting.detector = SURF::create(setting.minHessian);
  setting.showMatches = false;
  setting.detector = SIFT::create();
  return setting;
}