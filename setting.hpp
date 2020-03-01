#include <fstream>
#include "opencv2/xfeatures2d.hpp"

using std::ifstream;
using namespace cv;
using namespace cv::xfeatures2d;

struct Setting {
  String rootDir;
  bool showMatches;
  bool showCameras;
  int minHessian;
  int startFrame;
  int displayPcdCycle;
  float matchingRatioThresh;
  float errorRatio;
  float reprojectionErrorThresh;
  float ignoreDistanceRatio;
  int validPnpThresh;
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
  fs["ignoreDistanceRatio"] >> setting.ignoreDistanceRatio;
  fs["reprojectionErrorThresh"] >> setting.reprojectionErrorThresh;
  setting.showMatches = static_cast<std::string>(fs["showMatches"]) == "true";
  setting.showCameras = static_cast<std::string>(fs["showCameras"]) == "true";
  fs["displayPcdCycle"] >> setting.displayPcdCycle;
  fs["validPnpThresh"] >> setting.validPnpThresh;
  setting.detector = SIFT::create();
  return setting;
}