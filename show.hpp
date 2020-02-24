#include <opencv2/opencv.hpp>

#include "data.hpp"
#include "enums.hpp"

using namespace cv;
using std::cout;
using std::endl;
using std::vector;

void showPoints(dataBase db, dataSet ds) {
  int N = 0;
  for (size_t i = 0; i < db.points.size(); i++) {
    if (db.points[i].status == VALID) {
      N++;
    }
  }
  vector<Vec3d> cloud_mat;
  vector<Vec3b> colorsToShow;
  int cnt = 0;
  for (size_t i = 0; i < db.points.size(); i++) {
    if (db.points[i].status == VALID) {
      if (norm(db.points[i].positionAbs) < 500) {
        cloud_mat.push_back(db.points[i].positionAbs);
      } else {
        cloud_mat.push_back(Vec3d(0.0, 0.0, 0.0));
      }
      colorsToShow.push_back(db.colors[i]);
      cnt++;
    }
  }

  cout << "total:" << N << "points" << endl;
  if (N == 0) return;
  String winname = "Viz Camera Pose";
  viz::Viz3d myWindow(winname);
  for (size_t i = 0; i < db.images.size(); i++) {
    Mat pose = db.images[i].cameraPose;
    Mat R, t;
    R = (Mat_<double>(3, 3) << pose.at<double>(0, 0), pose.at<double>(0, 1),
         pose.at<double>(0, 2), pose.at<double>(1, 0), pose.at<double>(1, 1),
         pose.at<double>(1, 2), pose.at<double>(2, 0), pose.at<double>(2, 1),
         pose.at<double>(2, 2));
    t = (Mat_<double>(3, 1) << pose.at<double>(0, 3), pose.at<double>(1, 3),
         -pose.at<double>(2, 3));

    // t = (Mat_<double>(3, 1) << pose.at<double>(0, 3), pose.at<double>(1, 3),
    //      -pose.at<double>(2, 3));

    viz::WCameraPosition cpw(-0.1);
    viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599), -0.1);
    std::string widgetPoseName = "CPW" + std::to_string(i);
    std::string widgetFrustumName = "CPW_FRUSTUM" + std::to_string(i);
    myWindow.showWidget(widgetPoseName, cpw, Affine3d(R, t));
    myWindow.showWidget(widgetFrustumName, cpw_frustum, Affine3d(R, t));
  }
  viz::WCloud wcloud(cloud_mat, colorsToShow);
  // viz::WCloud wcloud(cloud_mat, viz::Color::white());
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
    if (idx >= 0 && db.points[idx].status != NO_POSITION) {
      int x = (int)db.images[i].keyPoints[j].pt.x;
      int y = (int)db.images[i].keyPoints[j].pt.y;
      db.colors[idx] = img.at<Vec3b>(Point(x, y));
    }
  }
  return true;
}