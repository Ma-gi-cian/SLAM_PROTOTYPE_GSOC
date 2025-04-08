#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "feature_matcher.h"

namespace cv{
	namespace slam{

struct PoseEstimationResult{
	bool success = false;
	cv::Mat pose;
	cv::vector<uchar> inlierMask;
	double reprojectionError = 0.0;
	int numInliers = 0;
};

class PoseEstimator {
public:
	virtual ~PoseEstimator() = default;

	virtual PoseEstimationResult estimateRelativePose(
			const std::vector<cv::KeyPoint>& keypoints1,
			const std::vector<cv::KeyPoint>& keypoint2,
			const std::vector<cv::DMatch>& matches;
			const cv::Mat& cameraMatrix,
			const cv::Mat& distCoeffs = cv::Mat()
			) = 0;
	virtual PoseEstimationResult estimateAbsolutePose(
			const std::vector<cv::Point2f>& imagePoints,
			const std::vector<cv::Point3f>& objectPoints,
			const cv::Mat& cameraMatrix,
			const cv::Mat& distCoeffs = cv::Mat()
			) = 0;
};
}
}
