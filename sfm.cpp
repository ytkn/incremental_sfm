#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"

#include "data.hpp"
#include "enums.hpp"
#include "mat.hpp"
#include "setting.hpp"
#include "show.hpp"

using namespace cv;
using std::cout;
using std::endl;
using std::ifstream;
using std::vector;

vector<DMatch> matchKeyPoints(Mat descriptors1, Mat descriptors2,
                              float ratio_thresh) {
  Ptr<DescriptorMatcher> matcher =
      DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
  vector<vector<DMatch> > knn_matches;
  matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
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

dataBase init(Mat img1, Mat img2, dataSet &ds, Setting &setting) {
  Mat K = ds.K;
  Mat distortion = ds.distortion;
  dataBase db;
  vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  setting.detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
  setting.detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
  vector<DMatch> good_matches =
      matchKeyPoints(descriptors1, descriptors2, setting.matchingRatioThresh);

  if (setting.showMatches) {
    showMatches(img1, img2, keypoints1, keypoints2, good_matches);
  }
  vector<long> idx1(keypoints1.size(), -1), idx2(keypoints2.size(), -1);
  vector<Point2d> p1, p2;
  collectUndistortedPoints(good_matches, keypoints1, keypoints2, K, distortion,
                           p1, p2, idx1, idx2);
  Mat E, R, t, mask;
  E = findEssentialMat(p1, p2, K, RANSAC, 0.999,
                       setting.reprojectionErrorThresh, mask);
  recoverPose(E, p1, p2, K, R, t, mask);
  Mat Rt1 = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
  Mat P1 = K * Rt1;
  Mat Rt2 = convertToRt(R, t);
  Mat P2 = K * Rt2;
  Mat result;
  triangulatePoints(P1, P2, p1, p2, result);
  db.images.push_back(image(descriptors1, keypoints1, idx1, Rt1));
  db.images.push_back(image(descriptors2, keypoints2, idx2, Rt2));
  for (int i = 0; i < result.cols; i++) {
    Vec3d pt = getPointIn3d(result, i);
    if ((int)mask.at<uchar>(i) >= 1) {
      point p(TENTATIVE, pt);
      db.points.push_back(p);
    } else {
      point p(NO_POSITION, pt);
      db.points.push_back(p);
    }
  }
  return db;
}

vector<long> registerAdditionalResult(dataBase &db, const Mat &result,
                                      const Mat &mask,
                                      const vector<DMatch> good_matches,
                                      const int prevIdx,
                                      const int curKeypointsNum,
                                      const Setting &setting) {
  vector<long> curIdx(curKeypointsNum, -1);
  const int curPoints = db.points.size();
  int cntNewPoints = 0;
  for (int i = 0; i < result.cols; i++) {
    Vec3d pt = getPointIn3d(result, i);
    int idx = db.images[prevIdx].keyPointIdx[good_matches[i].queryIdx];
    int idxCurKeypoint = good_matches[i].trainIdx;
    if ((int)mask.at<uchar>(i) >= 1) {
      if (idx == -1) {
        curIdx[idxCurKeypoint] = curPoints + cntNewPoints;
        cntNewPoints++;
        point p(TENTATIVE, pt);
        db.points.push_back(p);
      } else {
        // 登録済みの番号を入力
        curIdx[idxCurKeypoint] = idx;
        if (db.points[idx].status == TENTATIVE ||
            db.points[idx].status == VALID) {
          if (distance(db.points[idx].positionAbs, pt) >
              norm(pt) * setting.errorRatio) {
            db.points[idx].status = NO_POSITION;
          } else {
            db.points[idx].status = VALID;
            db.points[idx].positionAbs = pt;
          }
        }
        // else -> 何もしない
      }
      // outliers
    } else {
      if (idx == -1) {
        point p(NO_POSITION, pt);
        curIdx[idxCurKeypoint] = curPoints + cntNewPoints;
        cntNewPoints++;
        db.points.push_back(p);
      } else {
        curIdx[idxCurKeypoint] = idx;
        db.points[idx].status = NO_POSITION;
      }
    }
  }
  return curIdx;
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
  vector<DMatch> good_matches = matchKeyPoints(prevDescriptors, curDescriptors,
                                               setting.matchingRatioThresh);
  if (setting.showMatches) {
    Mat prevImage = ds.getImage(prevIdx + setting.startFrame);
    showMatches(prevImage, img, prevKeypoints, curKeypoints, good_matches);
  }
  int cnt = 0;
  for (size_t i = 0; i < good_matches.size(); i++) {
    int idx = db.images[prevIdx].keyPointIdx[good_matches[i].queryIdx];
    if (idx >= 0 && db.points[idx].status != NO_POSITION) {
      cnt++;
    }
  }
  cout << "matches:" << cnt << endl;
  int cur = 0;
  Mat pointAbs(cnt, 1, CV_32FC3);
  Mat pointInImage(cnt, 1, CV_32FC2);
  for (size_t i = 0; i < good_matches.size(); i++) {
    int idx = db.images[prevIdx].keyPointIdx[good_matches[i].queryIdx];
    if (idx >= 0 && db.points[idx].status != NO_POSITION) {
      pointAbs.at<Vec3f>(cur, 0) = db.points[idx].positionAbs;
      Vec2f pt;
      pt(0) = curKeypoints[good_matches[i].trainIdx].pt.x;
      pt(1) = curKeypoints[good_matches[i].trainIdx].pt.y;
      pointInImage.at<Vec2f>(cur, 0) = pt;
      cur++;
    }
  }

  Mat t, mask, Rvec, inliers, R;
  solvePnPRansac(pointAbs, pointInImage, K, distortion, Rvec, t, false, 1000,
                 0.4, 0.999, inliers);
  Rodrigues(Rvec, R);
  cout << R << endl;
  cout << t << endl;
  Mat P1 = K * db.images[prevIdx].cameraPose;
  Mat Rt2 = convertToRt(R, t);
  Mat P2 = K * Rt2;

  vector<Point2d> p1, p2;
  collectUndistortedPoints(good_matches, prevKeypoints, curKeypoints, K,
                           distortion, p1, p2);
  findEssentialMat(p1, p2, K, RANSAC, 0.999, setting.reprojectionErrorThresh,
                   mask);
  Mat result;
  triangulatePoints(P1, P2, p1, p2, result);
  vector<long> curIdx = registerAdditionalResult(
      db, result, mask, good_matches, prevIdx, curKeypoints.size(), setting);
  db.images.push_back(image(curDescriptors, curKeypoints, curIdx, Rt2));
  return true;
}

int main(int argc, char *argv[]) {
  Setting setting = initSetting("setting.yml");
  dataSet ds(setting.rootDir);
  cout << "Read image" << endl;
  Mat prev = ds.getImage(setting.startFrame);
  Mat cur = ds.getImage(setting.startFrame + 1);
  cout << prev.size() << endl;
  dataBase db = init(prev, cur, ds, setting);
  db.imageIdx.push_back(setting.startFrame);
  db.imageIdx.push_back(setting.startFrame + 1);
  addColor(ds, db, 1);
  // showPoints(db, ds);
  prev = cur;
  for (int i = setting.startFrame + 2; i < ds.numImages; i++) {
    cout << "Frame:" << i << endl;
    Mat cur = ds.getImage(i);
    addNewImage(db, i - setting.startFrame - 1, cur, ds, setting);
    addColor(ds, db, i - setting.startFrame - 1);
    db.imageIdx.push_back(i);
    if (i % setting.displayPcdCycle == 0 || i + 1 == ds.numImages)
      showPoints(db, ds);
  }
  return 0;
}