/******************************************************************************
 * Copyright (c) 2020, 郑州金惠-机器视觉事业部
 *
 * Project		: OpenStitch
 * Purpose		: 柱面投影类
 * Author		: 崔继祥
 * Created		: 2021-07-20
 * Modified by	: Cui Jixiang(崔继祥), cui_20151107@126.com
 * Modified     : 1.完成柱面投影的测试，并没有计算比例因子		2021-07-29
******************************************************************************/
#ifndef CYLINDER_WARP_H
#define CYLINDER_WARP_H

#include <opencv2/core.hpp>
#include <vector>

class CylinderWarp
{
public:
	CylinderWarp(const float& fFactor, const cv::Size& ImgSize, const float& fFocal);
	~CylinderWarp();
	///图像柱面投影
	cv::Mat	SoImgCylinderWarp(const cv::Mat& SrcImg);

	///匹配点柱面投影
	void SoMatchPtsCylinderWarp(std::vector<cv::KeyPoint>& vSrcPts, std::vector<cv::KeyPoint>& vCylinderPts);

	cv::Size SoGetCylinderSize();

	///单点逆投影
	cv::Point2f SoPtCylinderProjInv(const cv::Point2f& Pt);

private:

	///单点正投影
	cv::Point2f PtCylinderProj(const cv::Point2f& Pt);

	///单点逆投影
	//cv::Point2f PtCylinderProjInv(const cv::Point2f& Pt);

	///双线性内插
	cv::Point3f interpolate(const cv::Mat& Img, const cv::Point2f& Pano2ImgUV);

	cv::Size CalCylinderSize(const float& fFocal);

private:

	float				_fFactor;

	float				_fFocal;

	cv::Size			_ImgSize;

	cv::Size			_CylinderSize;
};
#endif		/*!< CYLINDER_WARP_H */
