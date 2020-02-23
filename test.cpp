#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

#define DEBUG true

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
using std::ifstream;
using std::vector;

class dataSet {
 public:
  int numImages;
  String rootDir;
  vector<String> filenames;
  Mat K;
  Mat distortion;
  dataSet(String root) {
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
    for (int i = 0; i < numImages; i++) {
      String fileName;
      imageNames >> fileName;
      filenames.push_back(fileName);
    }
    if (!dist) {
      distortion = (cv::Mat_<double>(4, 1) << 0, 0, 0, 0);
    } else {
      double k1, k2, p1, p2;
      dist >> k1 >> k2 >> p1 >> p2;
      distortion = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);
    }
  }
  Mat getImage(int n) {
    Mat img = imread(rootDir + filenames[n], IMREAD_GRAYSCALE);
    return img;
  }
  Mat getColorImage(int n) {
    Mat img = imread(rootDir + filenames[n], IMREAD_COLOR);
    return img;
  }
};

struct Setting {
  String rootDir;
  int minHessian;
  int start;
  float ratioThresh;
  Ptr<Feature2D> detector;
};

Setting initSetting(String filename) {
  Setting setting;
  ifstream settingFile(filename);
  settingFile >> setting.rootDir;
  settingFile >> setting.minHessian;
  settingFile >> setting.start;
  settingFile >> setting.ratioThresh;
  // setting.detector = SURF::create(setting.minHessian);
  setting.detector = SIFT::create();
  return setting;
}

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
  bool hasPosition;
  Vec3d positionAbs;
  point(bool hasPos, Vec3d pos) {
    hasPosition = hasPos;
    positionAbs = pos;
  }
};

class dataBase {
 public:
  vector<point> points;
  vector<image> images;
  vector<int> imageIdx;
  vector<Vec3b> colors;
  dataBase() {}
};

double norm(Vec3d p) { return sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]); }
double distance(Vec3d p1, Vec3d p2) {
  return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) +
              (p1[1] - p2[1]) * (p1[1] - p2[1]) +
              (p1[2] - p2[2]) * (p1[2] - p2[2]));
}

Mat convertToRt(Mat R, Mat t) {
  Mat Rt = (Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1),
            R.at<double>(0, 2), t.at<double>(0), R.at<double>(1, 0),
            R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
            t.at<double>(2));
  return Rt;
}

void showPoints(dataBase db, dataSet ds) {
  int N = 0;
  for (size_t i = 0; i < db.points.size(); i++) {
    if (db.points[i].hasPosition) {
      N++;
    }
  }
  vector<Vec3d> cloud_mat;
  vector<Vec3b> colorsToShow;
  int cnt = 0;
  for (size_t i = 0; i < db.points.size(); i++) {
    if (db.points[i].hasPosition) {
      if (norm(db.points[i].positionAbs) < 100) {
        cloud_mat.push_back(db.points[i].positionAbs);
      } else {
        cloud_mat.push_back(Vec3d(0.0, 0.0, 0.0));
      }
      colorsToShow.push_back(db.colors[i]);
      cnt++;
    }
  }

  cout << "total:" << N << "points" << endl;
  String winname = "Viz Camera Pose";
  viz::Viz3d myWindow(winname);
  for (size_t i = 0; i < db.images.size(); i++) {
    Mat pose = db.images[i].cameraPose;
    Mat R, t;
    R = (Mat_<double>(3, 3) << pose.at<double>(0, 0), pose.at<double>(0, 1),
         pose.at<double>(0, 2), pose.at<double>(1, 0), pose.at<double>(1, 1),
         pose.at<double>(1, 2), pose.at<double>(2, 0), pose.at<double>(2, 1),
         pose.at<double>(2, 2));
    t = (Mat_<double>(3, 1) << -pose.at<double>(0, 3), pose.at<double>(1, 3),
         pose.at<double>(2, 3));
    viz::WCameraPosition cpw(-0.1);
    viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599), -0.1);
    std::string widgetPoseName = "CPW" + std::to_string(i);
    std::string widgetFrustumName = "CPW_FRUSTUM" + std::to_string(i);
    myWindow.showWidget(widgetPoseName, cpw, Affine3d(R, t));
    myWindow.showWidget(widgetFrustumName, cpw_frustum, Affine3d(R, t));
  }
  viz::WCloud wcloud(cloud_mat, colorsToShow);
  myWindow.showWidget("Cloud", wcloud);
  myWindow.spin();
}

bool showMatches(Mat img1, Mat img2, vector<KeyPoint> keypoints1,
                 vector<KeyPoint> keypoints2, vector<DMatch> good_matches) {
  Mat img_matches;
  drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
              Scalar::all(-1), Scalar::all(-1), vector<char>(),
              DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  imshow("Matches", img_matches);
  waitKey(0);
  destroyWindow("Matches");
  return true;
}

bool addColor(dataSet ds, dataBase &db, int i) {
  Mat img = ds.getColorImage(db.imageIdx[i]);
  db.colors.resize(db.points.size());
  for (size_t j = 0; j < db.images[i].keyPointIdx.size(); j++) {
    int idx = db.images[i].keyPointIdx[j];
    if (idx >= 0 && db.points[idx].hasPosition) {
      int x = (int)db.images[i].keyPoints[j].pt.x;
      int y = (int)db.images[i].keyPoints[j].pt.y;
      db.colors[idx] = img.at<Vec3b>(Point(x, y));
    }
  }
  return true;
}

vector<DMatch> matchKeyPoints(Mat descriptors1, Mat descriptors2,
                              float ratio_thresh) {
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  vector<vector<DMatch> > knn_matches;
  cout << "Match start" << endl;
  cout << descriptors1.size() << ' ' << descriptors2.size() << endl;
  matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
  cout << "Match end" << endl;
  vector<DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance <
        ratio_thresh * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  return good_matches;
}

bool collectUndistortedPoints(const vector<DMatch> &good_matches,
                              const vector<KeyPoint> &keypoints1,
                              const vector<KeyPoint> &keypoints2, const Mat &K,
                              const Mat &distortion, vector<Point2d> &p1,
                              vector<Point2d> &p2, vector<long> &idx1,
                              vector<long> &idx2) {
  vector<Point2d> p1_, p2_;
  for (size_t i = 0; i < good_matches.size(); i++) {
    p1_.push_back(keypoints1[good_matches[i].queryIdx].pt);
    p2_.push_back(keypoints2[good_matches[i].trainIdx].pt);
    idx1[good_matches[i].queryIdx] = i;
    idx2[good_matches[i].trainIdx] = i;
  }
  undistortPoints(p1_, p1, K, distortion, noArray(), K);
  undistortPoints(p2_, p2, K, distortion, noArray(), K);
  return true;
}

bool collectUndistortedPoints(const vector<DMatch> &good_matches,
                              const vector<KeyPoint> &prevKeypoints,
                              const vector<KeyPoint> &curKeypoints, Mat &K,
                              const Mat &distortion, vector<Point2d> &p1,
                              vector<Point2d> &p2) {
  vector<Point2d> p1_, p2_;
  for (size_t i = 0; i < good_matches.size(); i++) {
    p1_.push_back(prevKeypoints[good_matches[i].queryIdx].pt);
    p2_.push_back(curKeypoints[good_matches[i].trainIdx].pt);
  }
  undistortPoints(p1_, p1, K, distortion, noArray(), K);
  undistortPoints(p2_, p2, K, distortion, noArray(), K);
  return true;
}

Vec3d getPointIn3d(const Mat points, const int idx) {
  Vec3d pt;
  pt(0) = points.at<double>(0, idx) / points.at<double>(3, idx);
  pt(1) = points.at<double>(1, idx) / points.at<double>(3, idx);
  pt(2) = points.at<double>(2, idx) / points.at<double>(3, idx);
  return pt;
}

dataBase init(Mat img1, Mat img2, dataSet ds, Setting setting) {
  Mat K = ds.K;
  Mat distortion = ds.distortion;
  dataBase db;
  vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  setting.detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
  setting.detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
  vector<DMatch> good_matches =
      matchKeyPoints(descriptors1, descriptors2, setting.ratioThresh);

  if (DEBUG) {
    showMatches(img1, img2, keypoints1, keypoints2, good_matches);
  }
  vector<long> idx1(keypoints1.size(), -1), idx2(keypoints2.size(), -1);
  vector<Point2d> p1, p2;
  collectUndistortedPoints(good_matches, keypoints1, keypoints2, K, distortion,
                           p1, p2, idx1, idx2);
  Mat E, R, t, mask;
  E = findEssentialMat(p1, p2, K, RANSAC, 0.999, 0.4, mask);
  recoverPose(E, p1, p2, K, R, t, mask);
  Mat Rt1 = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  Mat P1 = K * Rt1;
  Mat Rt2 = convertToRt(R, t);
  Mat P2 = K * Rt2;
  Mat result;
  triangulatePoints(P1, P2, p1, p2, result);
  Mat initialPos = (Mat_<double>(3, 1) << 0, 0, 0);
  db.images.push_back(image(descriptors1, keypoints1, idx1, Rt1));
  db.images.push_back(image(descriptors2, keypoints2, idx2, Rt2));
  for (int i = 0; i < result.cols; i++) {
    Vec3d pt = getPointIn3d(result, i);
    if ((int)mask.at<uchar>(i) >= 1) {
      point p(true, pt);
      db.points.push_back(p);
    } else {
      point p(false, pt);
      db.points.push_back(p);
    }
  }

  if (DEBUG) {
    cout << "POSE" << endl;
    cout << R << endl;
    cout << t << endl;
  }
  return db;
}

bool addNewImage(dataBase &db, int prevIdx, Mat img, dataSet ds,
                 Setting setting) {
  Mat K = ds.K;
  Mat distortion = ds.distortion;
  vector<KeyPoint> curKeypoints;
  vector<KeyPoint> prevKeypoints = db.images[prevIdx].keyPoints;
  Mat curDescriptors;
  Mat prevDescriptors = db.images[prevIdx].descriptors;
  setting.detector->detectAndCompute(img, noArray(), curKeypoints,
                                     curDescriptors);
  vector<DMatch> good_matches =
      matchKeyPoints(prevDescriptors, curDescriptors, setting.ratioThresh);
  if (DEBUG) {
    Mat prevImage = ds.getImage(prevIdx + setting.start);
    showMatches(prevImage, img, prevKeypoints, curKeypoints, good_matches);
  }
  int cnt = 0;
  for (size_t i = 0; i < good_matches.size(); i++) {
    int idx = db.images[prevIdx].keyPointIdx[good_matches[i].queryIdx];
    if (idx >= 0 && db.points[idx].hasPosition) {
      cnt++;
    }
  }
  cout << cnt << endl;
  int cur = 0;
  Mat pointAbs(cnt, 1, CV_32FC3);
  Mat pointInImage(cnt, 1, CV_32FC2);
  for (size_t i = 0; i < good_matches.size(); i++) {
    int idx = db.images[prevIdx].keyPointIdx[good_matches[i].queryIdx];
    if (idx >= 0 && db.points[idx].hasPosition) {
      pointAbs.at<Vec3f>(cur, 0) = db.points[idx].positionAbs;
      Vec2f pt;
      pt(0) = curKeypoints[good_matches[i].trainIdx].pt.x;
      pt(1) = curKeypoints[good_matches[i].trainIdx].pt.y;
      pointInImage.at<Vec2f>(cur, 0) = pt;
      cur++;
    }
  }

  Mat t, mask, Rvec, inliers;
  solvePnPRansac(pointAbs, pointInImage, K, distortion, Rvec, t, false, 1000,
                 0.4, 0.999, inliers);
  Mat R;
  Rodrigues(Rvec, R);
  cout << R << endl;
  cout << t << endl;
  Mat P1 = K * db.images[prevIdx].cameraPose;
  Mat Rt2 = convertToRt(R, t);
  Mat P2 = K * Rt2;

  vector<Point2d> p1, p2;
  collectUndistortedPoints(good_matches, prevKeypoints, curKeypoints, K,
                           distortion, p1, p2);
  findEssentialMat(p1, p2, K, RANSAC, 0.999, 0.4, mask);
  Mat result;
  triangulatePoints(P1, P2, p1, p2, result);
  int curPoints = db.points.size();
  int cntNewPoints = 0;
  vector<long> curIdx(curKeypoints.size(), -1);
  for (int i = 0; i < result.cols; i++) {
    Vec3d pt = getPointIn3d(result, i);
    int idx = db.images[prevIdx].keyPointIdx[good_matches[i].queryIdx];
    int idxCurKeypoint = good_matches[i].trainIdx;
    if ((int)mask.at<uchar>(i) >= 1) {
      point p(true, pt);
      if (idx == -1) {
        curIdx[idxCurKeypoint] = curPoints + cntNewPoints;
        cntNewPoints++;
        db.points.push_back(p);
      } else {
        // 登録済みの番号を入力
        curIdx[idxCurKeypoint] = idx;
        if (db.points[idx].hasPosition) {
          // 何もしない
        } else {
          // positionを更新
          db.points[idx].hasPosition = true;
          db.points[idx].positionAbs = pt;
        }
      }
    } else {
      point p(false, pt);
      if (idx == -1) {
        curIdx[idxCurKeypoint] = curPoints + cntNewPoints;
        cntNewPoints++;
        db.points.push_back(p);
      } else {
        curIdx[idxCurKeypoint] = idx;
        // もうすでに登録済みの場合は何もしない
      }
    }
  }

  db.images.push_back(image(curDescriptors, curKeypoints, curIdx, Rt2));
  return true;
}

int main(int argc, char *argv[]) {
  Setting setting = initSetting("setting.txt");
  dataSet ds(setting.rootDir);
  Mat prev = ds.getImage(setting.start);
  Mat cur = ds.getImage(setting.start + 1);
  cout << prev.size() << endl;
  dataBase db = init(prev, cur, ds, setting);
  db.imageIdx.push_back(setting.start);
  db.imageIdx.push_back(setting.start + 1);
  addColor(ds, db, 1);
  showPoints(db, ds);
  prev = cur;
  for (int i = setting.start + 2; i <= ds.numImages; i++) {
    cout << i << endl;
    Mat cur = ds.getImage(i);
    addNewImage(db, i - setting.start - 1, cur, ds, setting);
    addColor(ds, db, i - setting.start - 1);
    db.imageIdx.push_back(i);
    showPoints(db, ds);
  }
  return 0;
}