#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

namespace cv{
	namespace slam {

struct MatchResult {
	std::vector<cv::DMatch> matches;
	std::vector<cv::DMatch> rawMatches;
	std::vector<uchar> inlierMask;
};

class FeatureMatcher{
public:
	virtual ~FeatureMatcher() = default;

	virtual MatchResult match(
			const cv::Mat& descriptors1,
			const cv::Mat& descriptors2,
			const std::vector<cv::KeyPoint>& keypoints1,
			const std::vector<cv::KeyPoint>& keypoints2 ) = 0;

	virtual void setGeometricVerification(
			bool useGeometricVerification,
			double ransacThreshold = 4.0
			) = 0;
};

class BruteForceMatcher : public FeatureMatcher {
public:
	BruteForceMatcher(
			int descriptorType = CV_8U,
			int normType = cv::NORM_HAMMING,
			bool crossCheck = false,
			float ratioThreshold = 0.75f
			);

	MatchResult match(
			const cv::Mat& descriptors1,
			const cv::Mat& descriptors2,
			const std::vector<cv::KeyPoint>& keypoints1,
			const std::vector<cv::KeyPoint>& keypoints2 ) override;

	void setGeometricVerification(
			bool useGeometricVerification,
			double ransacThreshold = 4.0
			) override;
private:
	cv::Ptr<cv::BFMatcher> mMatcher;
	float mRatioThreshold;
	bool mUseGeometricVerification;
	double mRansacThreshold;
};
}
}

