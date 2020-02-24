#include <opencv2/opencv.hpp>

using namespace cv;

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

Vec3d getPointIn3d(const Mat points, const int idx) {
  Vec3d pt;
  pt(0) = points.at<double>(0, idx) / points.at<double>(3, idx);
  pt(1) = points.at<double>(1, idx) / points.at<double>(3, idx);
  pt(2) = points.at<double>(2, idx) / points.at<double>(3, idx);
  return pt;
}