#include <opencv2/calib3d.hpp>

#include <iostream>

#include "Camera.h"

namespace OpenStitch
{
	void Camera::SoWaveStraighten(std::vector<Camera>& vCamers)
	{
		///此处与opencv不同，opencv存放的相机旋转是此处的转置
		///波形矫正没有看明白，套用opencv源码 并做出修改

		for (auto& i : vCamers)
		{
			i.mR = i.mR.t();		/*!< 变成转置，在按照opencv的方式处理 */
		}

		cv::Mat ROT_RO = cv::Mat::zeros(3, 3, CV_64F);
		for (const auto& i : vCamers)
		{
			cv::Mat col = i.mR.col(0);
			ROT_RO += col * col.t();
		}
		cv::Mat eigen_vals, eigen_vecs;
		cv::eigen(ROT_RO, eigen_vals, eigen_vecs);

		cv::Mat q1 = eigen_vecs.row(2).t();

		cv::Mat R2 = cv::Mat::zeros(3, 1, CV_64F);
		for (const auto& i : vCamers)
		{
			R2 += i.mR.col(2);
		}

		//cv::Mat q0 = R2.cross(q1);
		cv::Mat q0 = q1.cross(R2);

		q0 /= cv::norm(q0);

		cv::Mat q2 = q0.cross(q1);

		double conf = 0;

		for (const auto& i : vCamers)
		{
			conf += q0.dot(i.mR.col(0));
		}
		if (conf < 0)
		{
			q0 *= -1;
			q1 *= -1;
		}

		cv::Mat Q = cv::Mat::zeros(3, 3, CV_64F);
		//Q.col(0) = q0.col(0); Q.col(1) = q1.col(0); Q.col(2) = q2.col(0);

		cv::Mat temp = Q.row(0);
		cv::Mat(q0.t()).copyTo(temp);
		temp = Q.row(1);
		cv::Mat(q1.t()).copyTo(temp);
		temp = Q.row(2);
		cv::Mat(q2.t()).copyTo(temp);

		//std::cout << "q0 : " << q0 << std::endl;
		//std::cout << "q1 : " << q1 << std::endl;
		//std::cout << "q2 : " << q2 << std::endl;

		for (auto& i : vCamers)
		{
			//std::cout << "Q: " << Q << std::endl;
			i.mR = (Q * i.mR).t();
		}
	}

	void Camera::SoR2so3(const cv::Mat& R, cv::Mat& so3)
	{
		cv::Rodrigues(R, so3);
	}

	void Camera::Soso32R(const cv::Mat& so3, cv::Mat& R)
	{
		cv::Rodrigues(so3, R);
	}

	cv::Mat Camera::SoGetK(const float& fFocal, const float& fCx, const float& fCy)
	{
		cv::Mat TempK = cv::Mat::eye(3, 3, CV_64F);
		TempK.at<double>(0, 0) = fFocal;
		TempK.at<double>(0, 2) = fCx;

		TempK.at<double>(1, 1) = fFocal;
		TempK.at<double>(1, 2) = fCy;
		return TempK;
	}
}