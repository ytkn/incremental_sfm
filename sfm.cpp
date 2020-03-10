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
#include "match.hpp"
#include "setting.hpp"
#include "show.hpp"

using namespace cv;
using std::cout;
using std::endl;
using std::ifstream;
using std::vector;

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

double meanFlowLength(const vector<DMatch> &good_matches,
                      const vector<KeyPoint> &keypoints1,
                      const vector<KeyPoint> &keypoints2) {
  double sum = 0;
  for (size_t i = 0; i < good_matches.size(); i++) {
    Point2d p1 = keypoints1[good_matches[i].queryIdx].pt;
    Point2d p2 = keypoints2[good_matches[i].trainIdx].pt;
    sum += sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
  }
  return sum / good_matches.size();
}

bool init(Mat img1, Mat img2, DataSet &ds, Setting &setting, DataBase &db) {
  Mat K = ds.K;
  Mat distortion = ds.distortion;
  vector<KeyPoint> keypoints1, keypoints2;
  Mat descriptors1, descriptors2;
  setting.detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
  setting.detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
  vector<DMatch> good_matches =
      matchKeyPoints(descriptors1, descriptors2, setting.matchingRatioThresh);
  if (meanFlowLength(good_matches, keypoints1, keypoints2) <
      setting.minFlowLength) {
    cout << "rejected" << endl;
    return false;
  }
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
  return true;
}

vector<long> registerAdditionalResult(DataBase &db, const Mat &result,
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

enum collectionCliretia { WITH_POS, WITH_VALID_POS };
bool isMatchCliteria(PointStatus status, collectionCliretia criteria) {
  if (criteria == WITH_POS)
    return status != NO_POSITION;
  else
    return status == VALID;
}

bool collectPointWithPosition(const vector<DMatch> &good_matches,
                              const DataBase &db, const int prevIdx,
                              const DataSet &ds, Setting setting,
                              const vector<KeyPoint> curKeypoints,
                              Mat &pointAbs, Mat &pointInImage) {
  int cntWithPos = 0;
  int cntWithValidPos = 0;
  for (size_t i = 0; i < good_matches.size(); i++) {
    int idx = db.images[prevIdx].keyPointIdx[good_matches[i].queryIdx];
    if (idx >= 0 && db.points[idx].status != NO_POSITION) cntWithPos++;
    if (idx >= 0 && db.points[idx].status == VALID) cntWithValidPos++;
  }
  cout << "with pos:" << cntWithPos << " with valid pos:" << cntWithValidPos
       << ' ' << setting.validPnpThresh << endl;
  collectionCliretia criteria =
      (cntWithValidPos >= setting.validPnpThresh ? WITH_VALID_POS : WITH_POS);
  int cnt = (cntWithValidPos >= setting.validPnpThresh ? cntWithValidPos
                                                       : cntWithPos);
  Mat pointAbs_(cnt, 1, CV_32FC3);
  Mat pointInImage_(cnt, 1, CV_32FC2);
  int cur = 0;
  for (size_t i = 0; i < good_matches.size(); i++) {
    int idx = db.images[prevIdx].keyPointIdx[good_matches[i].queryIdx];
    if (idx >= 0 && isMatchCliteria(db.points[idx].status, criteria)) {
      pointAbs_.at<Vec3f>(cur, 0) = db.points[idx].positionAbs;
      Vec2f pt;
      pt(0) = curKeypoints[good_matches[i].trainIdx].pt.x;
      pt(1) = curKeypoints[good_matches[i].trainIdx].pt.y;
      pointInImage_.at<Vec2f>(cur, 0) = pt;
      cur++;
    }
  }
  pointAbs = pointAbs_.clone();
  pointInImage = pointInImage_.clone();
  return true;
}

bool addNewImage(DataBase &db, const int prevIdx, Mat img, DataSet &ds,
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

  if (meanFlowLength(good_matches, prevKeypoints, curKeypoints) <
      setting.minFlowLength) {
    cout << "rejected" << endl;
    return false;
  }
  if (setting.showMatches) {
    Mat prevImage = ds.getImage(prevIdx + setting.startFrame);
    showMatches(prevImage, img, prevKeypoints, curKeypoints, good_matches);
  }
  Mat pointAbs, pointInImage;
  collectPointWithPosition(good_matches, db, prevIdx, ds, setting, curKeypoints,
                           pointAbs, pointInImage);
  Mat t, mask, Rvec, inliers, R;
  solvePnPRansac(pointAbs, pointInImage, K, distortion, Rvec, t, false, 1000,
                 setting.reprojectionErrorThresh, 0.999, inliers);
  Rodrigues(Rvec, R);
  cout << R << t << endl;
  Mat P1 = K * db.images[prevIdx].cameraPose;
  Mat Rt2 = convertToRt(R, t);
  Vec3d prevPosition = RtToPosition(db.images[prevIdx].cameraPose);
  Vec3d curPosition = RtToPosition(Rt2);
  if (distance(prevPosition, curPosition) <
      setting.ignoreDistanceRatio * db.initialDistance()) {
    cout << "rejected" << endl;
    return false;
  }
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

bool rematchImage(const DataBase &db, const int targetIdx, Mat img, DataSet &ds,
                  Setting setting) {
  Mat K = ds.K;
  Mat distortion = ds.distortion;
  vector<KeyPoint> curKeypoints;
  vector<KeyPoint> prevKeypoints = db.images[targetIdx].keyPoints;
  Mat curDescriptors;
  Mat prevDescriptors = db.images[targetIdx].descriptors;
  setting.detector->detectAndCompute(img, noArray(), curKeypoints,
                                     curDescriptors);
  vector<DMatch> good_matches = matchKeyPoints(prevDescriptors, curDescriptors,
                                               setting.matchingRatioThresh);
  if (good_matches.size() < 100) {
    return false;
  }
  if (true) {
    Mat prevImage = ds.getImage(targetIdx + setting.startFrame);
    showMatches(prevImage, img, prevKeypoints, curKeypoints, good_matches);
  }
  vector<Point2d> p1, p2;
  collectUndistortedPoints(good_matches, prevKeypoints, curKeypoints, K,
                           distortion, p1, p2);
  Mat E, R, t, mask;
  E = findEssentialMat(p1, p2, K, RANSAC, 0.999,
                       setting.reprojectionErrorThresh, mask);
  recoverPose(E, p1, p2, K, R, t, mask);
  cout << "detected loop with " << targetIdx << endl;
  cout << "relative position is " << R << t << endl;
  return true;
}

int main(int argc, char *argv[]) {
  Setting setting = initSetting("setting.yml");
  DataSet ds(setting.rootDir);
  cout << "Read image" << endl;
  Mat prev = ds.getImage(setting.startFrame);
  int curImgIdx = 1;
  Mat cur = ds.getImage(setting.startFrame + curImgIdx);
  cout << prev.size() << endl;
  DataBase db;
  LoopDetector loopDetector("small_voc.yml.gz");
  while (!init(prev, cur, ds, setting, db)) {
    curImgIdx++;
    cur = ds.getImage(setting.startFrame + curImgIdx);
  }
  loopDetector.addImageAndDetectLoop(prev);
  loopDetector.addImageAndDetectLoop(cur);
  db.imageIdx.push_back(setting.startFrame);
  db.imageIdx.push_back(setting.startFrame + curImgIdx);
  addColor(ds, db, curImgIdx);
  showPoints(db, ds, setting.showCameras);
  prev = cur;
  for (int i = setting.startFrame + curImgIdx + 1; i < ds.numImages; i++) {
    cout << "Frame:" << i << endl;
    Mat cur = ds.getImage(i);
    int prevIdx = db.lastIdx();
    cout << prevIdx << endl;
    bool added = addNewImage(db, prevIdx, cur, ds, setting);
    if (added) {
      int matchIdx = loopDetector.addImageAndDetectLoop(cur);
      if (matchIdx >= 0) {
        rematchImage(db, matchIdx, cur, ds, setting);
      }
      addColor(ds, db, prevIdx);
    }
    db.imageIdx.push_back(i);
    if (i % setting.displayPcdCycle == 0 || i + 1 == ds.numImages)
      showPoints(db, ds, setting.showCameras);
  }
  return 0;
}