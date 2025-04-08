#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

namespace cv {
	namespace slam {

// Class for implementaion of feature extractor
class FeatureExtractor {
public:
	virtual ~FeatureExtractor() = default;

	virtual int extract(
			const cv::Mat& image,
			std::vector<cv::KeyPoint>& keypoints,
			cv::Mat& descriptors
			) = 0;
	virtual void setROI(const cv::Rect& roi) = 0;
}

class OrbExtractor : public FeatureExtractor {
public:
	OrbExtractor(
			int nFeatures = 1000,
			float scaleFactor = 1.2f,
			int nLevels = 8
		    );
	int extract(
			const cv::Mat& image,
			std::vector<cv::KeyPoint>& keypoints,
			cv::Mat& descriptors
		   ) override;
	void setROI(const cv::Rect& roi) override;

private:
	cv::Ptr<cv::ORB> mOrb;
	cv::Rect mROI;
};
}
}


