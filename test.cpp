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

class dataSet {
 public:
  int numImages;
  String rootDir;
  std::vector<String> filenames;
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

int minHessian = 800;

class image {
 public:
  Mat descriptors;
  std::vector<KeyPoint> keyPoints;
  std::vector<long> keyPointIdx;
  Mat cameraPose;
  image(Mat ds, std::vector<KeyPoint> ks, std::vector<long> kpis, Mat cps) {
    descriptors = ds;
    keyPoints = std::vector<KeyPoint>(ks.size());
    for (size_t i = 0; i < ks.size(); i++) keyPoints[i] = ks[i];
    keyPointIdx = std::vector<long>(kpis.size());
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
  std::vector<point> points;
  std::vector<image> images;
  std::vector<int> imageIdx;
  std::vector<Vec3b> colors;
  dataBase() {}
};

void showPoints(dataBase db, dataSet ds) {
  int N = 0;
  for (size_t i = 0; i < db.points.size(); i++) {
    if (db.points[i].hasPosition) {
      N++;
    }
  }
  std::vector<Vec3d> cloud_mat;
  std::vector<Vec3b> colorsToShow;
  int cnt = 0;
  for (size_t i = 0; i < db.points.size(); i++) {
    if (db.points[i].hasPosition) {
      cloud_mat.push_back(db.points[i].positionAbs);
      colorsToShow.push_back(db.colors[i]);
      cnt++;
    }
  }
  cout << "total:" << N << "points" << endl;
  String winname = "Viz Camera Pose";
  viz::Viz3d myWindow(winname);
  viz::WCloud wcloud(cloud_mat, colorsToShow);
  myWindow.showWidget("Cloud", wcloud);
  myWindow.spin();
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

bool matchKeyPoints() { return true; }

dataBase init(Mat img1, Mat img2, dataSet ds) {
  std::vector<Point2d> p1_;
  std::vector<Point2d> p2_;
  Mat K = ds.K;
  Mat distortion = ds.distortion;
  dataBase db;
  Ptr<SURF> detector = SURF::create(minHessian);
  std::vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
  detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  std::vector<std::vector<DMatch> > knn_matches;
  matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

  const float ratio_thresh = 0.7f;
  std::vector<DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance <
        ratio_thresh * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  if (false) {
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                Scalar::all(-1), Scalar::all(-1), std::vector<char>(),
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("init Matches", img_matches);
    waitKey(0);
  }
  std::vector<long> idx1(keypoints1.size(), -1), idx2(keypoints2.size(), -1);
  for (size_t i = 0; i < good_matches.size(); i++) {
    p1_.push_back(keypoints1[good_matches[i].queryIdx].pt);
    p2_.push_back(keypoints2[good_matches[i].trainIdx].pt);
    idx1[good_matches[i].queryIdx] = i;
    idx2[good_matches[i].trainIdx] = i;
  }
  std::vector<Point2d> p1;
  std::vector<Point2d> p2;
  undistortPoints(p1_, p1, K, distortion, noArray(), K);
  undistortPoints(p2_, p2, K, distortion, noArray(), K);
  Mat E, R, t, mask;
  E = findEssentialMat(p1, p2, K, RANSAC, 0.999, 0.4, mask);
  recoverPose(E, p1, p2, K, R, t, mask);
  Mat Rt1 = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  Mat P1 = K * Rt1;
  Mat Rt2 = (Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1),
             R.at<double>(0, 2), t.at<double>(0), R.at<double>(1, 0),
             R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
             R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
             t.at<double>(2));
  Mat P2 = K * Rt2;
  Mat result;
  triangulatePoints(P1, P2, p1, p2, result);
  Mat initialPos = (Mat_<double>(3, 1) << 0, 0, 0);
  db.images.push_back(image(descriptors1, keypoints1, idx1, Rt1));
  db.images.push_back(image(descriptors2, keypoints2, idx2, Rt2));
  for (int i = 0; i < result.cols; i++) {
    Vec3d pt;
    pt(0) = result.at<double>(0, i) / result.at<double>(3, i);
    pt(1) = result.at<double>(1, i) / result.at<double>(3, i);
    pt(2) = result.at<double>(2, i) / result.at<double>(3, i);
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

bool addNewImage(dataBase &db, int prevIdx, Mat img, dataSet ds) {
  Mat K = ds.K;
  Mat distortion = ds.distortion;
  Ptr<SURF> detector = SURF::create(minHessian);
  std::vector<KeyPoint> curKeypoints;
  std::vector<KeyPoint> prevKeypoints = db.images[prevIdx].keyPoints;
  Mat curDescriptors;
  Mat prevDescriptors = db.images[prevIdx].descriptors;
  detector->detectAndCompute(img, noArray(), curKeypoints, curDescriptors);
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  std::vector<std::vector<DMatch> > knn_matches;
  matcher->knnMatch(prevDescriptors, curDescriptors, knn_matches, 2);
  cout << "match" << endl;
  const float ratio_thresh = 0.7f;
  std::vector<DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i][0].distance <
        ratio_thresh * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  cout << "matched" << endl;
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
      Vec2f pt, pt_;
      pt(0) = curKeypoints[good_matches[i].trainIdx].pt.x;
      pt(1) = curKeypoints[good_matches[i].trainIdx].pt.y;
      pt_(0) = prevKeypoints[good_matches[i].queryIdx].pt.x;
      pt_(1) = prevKeypoints[good_matches[i].queryIdx].pt.y;
      pointInImage.at<Vec2f>(cur, 0) = pt;
      cur++;
    }
  }

  Mat t, mask;
  Mat Rvec, inliers;
  solvePnPRansac(pointAbs, pointInImage, K, distortion, Rvec, t, false, 1000,
                 0.4, 0.999, inliers);
  Mat R;
  Rodrigues(Rvec, R);
  cout << R << endl;
  cout << t << endl;
  Mat P1 = K * db.images[prevIdx].cameraPose;
  Mat Rt2 = (cv::Mat_<double>(3, 4) << R.at<double>(0, 0), R.at<double>(0, 1),
             R.at<double>(0, 2), t.at<double>(0), R.at<double>(1, 0),
             R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1),
             R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
             t.at<double>(2));
  Mat P2 = K * Rt2;
  // calc position

  std::vector<Point2d> p1_, p2_;
  for (size_t i = 0; i < good_matches.size(); i++) {
    p1_.push_back(prevKeypoints[good_matches[i].queryIdx].pt);
    p2_.push_back(curKeypoints[good_matches[i].trainIdx].pt);
  }
  std::vector<Point2d> p1;
  std::vector<Point2d> p2;
  undistortPoints(p1_, p1, K, distortion, noArray(), K);
  undistortPoints(p2_, p2, K, distortion, noArray(), K);
  findEssentialMat(p1, p2, K, RANSAC, 0.999, 0.4, mask);
  Mat result;
  triangulatePoints(P1, P2, p1, p2, result);
  int curPoints = db.points.size();
  int cntNewPoints = 0;
  std::vector<long> curIdx(curKeypoints.size(), -1);
  for (int i = 0; i < result.cols; i++) {
    Vec3d pt;
    int idx = db.images[prevIdx].keyPointIdx[good_matches[i].queryIdx];
    int idxCurKeypoint = good_matches[i].trainIdx;
    pt(0) = result.at<double>(0, i) / result.at<double>(3, i);
    pt(1) = result.at<double>(1, i) / result.at<double>(3, i);
    pt(2) = result.at<double>(2, i) / result.at<double>(3, i);
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
const String rootDir =
    "/home/yuta/work/study/ImageProcessing/dateasets/temple/";
// "/home/yuta/work/study/ImageProcessing/dateasets/ImageDataset_SceauxCastle/"
// "images/";
// "/home/yuta/work/study/ImageProcessing/dateasets/gerrard-hall/images/";

int main(int argc, char *argv[]) {
  dataSet ds(rootDir);
  int start = 50;
  Mat prev = ds.getImage(start);
  Mat cur = ds.getImage(start + 1);
  dataBase db = init(prev, cur, ds);
  db.imageIdx.push_back(start);
  db.imageIdx.push_back(start + 1);
  addColor(ds, db, 1);
  showPoints(db, ds);
  prev = cur;
  for (int i = start + 2; i <= ds.numImages; i++) {
    cout << i << endl;
    Mat cur = ds.getImage(i);
    cout << cur.size << endl;
    addNewImage(db, i - start - 1, cur, ds);
    addColor(ds, db, i - start - 1);
    db.imageIdx.push_back(i);
    cout << "show" << endl;
    showPoints(db, ds);
  }
  return 0;
}