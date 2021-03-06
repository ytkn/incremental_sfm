#ifndef INCLUDED_Data_h_
#define INCLUDED_Data_h_

#include <fstream>
#include <opencv2/opencv.hpp>
#include <queue>
#include "enums.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "DBoW2.h"

using std::ifstream;
using namespace cv;

using std::cout;
using std::endl;
using std::queue;
using std::vector;

class DataSet {
 public:
  int numImages;
  String rootDir;
  vector<String> filenames;
  Mat K;
  Mat distortion;
  DataSet(String root) {
    rootDir = root;
    cout << root << endl;
    ifstream calib(rootDir + "K.txt");
    ifstream imageNames(rootDir + "images.txt");
    ifstream dist(rootDir + "distortion.txt");
    double k11, k12, k13, k21, k22, k23, k31, k32, k33;
    calib >> k11 >> k12 >> k13 >> k21 >> k22 >> k23 >> k31 >> k32 >> k33;
    K = (cv::Mat_<double>(3, 3) << k11, k12, k13, k21, k22, k23, k31, k32, k33);
    imageNames >> numImages;
    cout << K << endl;
    cout << "Num images:" << numImages << endl;
    for (int i = 0; i < numImages; i++) {
      String fileName;
      imageNames >> fileName;
      filenames.push_back(fileName);
    }
    if (!dist) {
      distortion = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);
    } else {
      double k1, k2, p1, p2, k3;
      dist >> k1 >> k2 >> p1 >> p2 >> k3;
      distortion = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
    }
    cout << distortion << endl;
    cout << "initialized dataset" << endl;
  }
  Mat getImage(int n) {
    cout << "read " << filenames[n] << endl;
    Mat img = imread(rootDir + filenames[n], IMREAD_GRAYSCALE);
    return img;
  }
  Mat getColorImage(int n) {
    Mat img = imread(rootDir + filenames[n], IMREAD_COLOR);
    return img;
  }
};

class image {
 public:
  Mat descriptors;
  vector<KeyPoint> keyPoints;
  vector<long> keyPointIdx;
  Mat cameraPose;
  image(Mat ds, vector<KeyPoint> ks, vector<long> kpis, Mat cps) {
    descriptors = ds;
    keyPoints = vector<KeyPoint>(ks.size());
    for (size_t i = 0; i < ks.size(); i++) keyPoints[i] = ks[i];
    keyPointIdx = vector<long>(kpis.size());
    for (size_t i = 0; i < kpis.size(); i++) keyPointIdx[i] = kpis[i];
    cameraPose = cps;
  }
};

class point {
 public:
  PointStatus status;
  Vec3d positionAbs;
  point(PointStatus _status, Vec3d pos) {
    status = _status;
    positionAbs = pos;
  }
};

class DataBase {
 public:
  vector<point> points;
  vector<image> images;
  vector<int> imageIdx;
  vector<Vec3b> colors;
  DataBase() {}
  int lastIdx() { return images.size() - 1; }
  double initialDistance() {
    Mat Rt = images[1].cameraPose;
    Vec3d t =
        Vec3d(Rt.at<double>(0, 3), Rt.at<double>(1, 3), Rt.at<double>(2, 3));
    cout << t << ' ';
    return norm(t);
  }
};

class LoopDetector {
 public:
  size_t loopInterval;
  OrbDatabase db;
  Ptr<ORB> orb;
  queue<vector<Mat>> fque;
  LoopDetector(String vocFile) {
    loopInterval = 50;
    orb = ORB::create();
    OrbVocabulary voc(vocFile);
    OrbDatabase db_(voc, false, 0);
    db = db_;
  }
  int addImageAndDetectLoop(Mat image) {
    vector<KeyPoint> keypoints;
    Mat mask;
    Mat descriptors;
    orb->detectAndCompute(image, mask, keypoints, descriptors);
    vector<Mat> desc;
    changeStructure(descriptors, desc);
    int ret = detectLoop(desc);
    fque.push(desc);
    if (fque.size() > loopInterval) {
      db.add(fque.front());
      fque.pop();
    }
    return ret;
  }

 private:
  int detectLoop(vector<Mat> desc) {
    if (db.size() == 0) return -1;
    DBoW2::QueryResults ret;
    db.query(desc, ret, 1);
    return (ret[0].Score > 0.45 ? ret[0].Id - 1 : -1);
  }

  void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out) {
    out.resize(plain.rows);

    for (int i = 0; i < plain.rows; ++i) {
      out[i] = plain.row(i);
    }
  }
};

#endif