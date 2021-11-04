#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/core.hpp>
#include <vector>

namespace OpenStitch
{
	class Camera
	{
	public:

		static void SoWaveStraighten(std::vector<Camera>& vCamers);
		static void SoR2so3(const cv::Mat& R, cv::Mat& so3);
		static void Soso32R(const cv::Mat& so3, cv::Mat& R);
		static cv::Mat SoGetK(const float& fFocal, const float& fCx, const float& fCy);

	public:

		cv::Mat					mR;
		cv::Mat					mK;
		float					mfFocal;
		float					mfCx, mfCy;
	};
}

#endif		/*!< CAMERA_H */