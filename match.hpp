#ifndef INCLUDED_Match_h_
#define INCLUDED_Match_h_

#include <opencv2/opencv.hpp>

using namespace cv;
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

#endif