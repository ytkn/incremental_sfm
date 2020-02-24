#include <fstream>
#include "opencv2/xfeatures2d.hpp"

using std::ifstream;
using namespace cv;
using namespace cv::xfeatures2d;

struct Setting {
  String rootDir;
  int minHessian;
  int start;
  float ratioThresh;
  float errorRatio;
  float reprojectionErrorThreshold;
  Ptr<Feature2D> detector;
};

Setting initSetting(String filename) {
  Setting setting;
  ifstream settingFile(filename);
  settingFile >> setting.rootDir;
  settingFile >> setting.minHessian;
  settingFile >> setting.start;
  settingFile >> setting.ratioThresh;
  settingFile >> setting.errorRatio;
  settingFile >> setting.reprojectionErrorThreshold;
  // setting.detector = SURF::create(setting.minHessian);
  setting.detector = SIFT::create();
  return setting;
}