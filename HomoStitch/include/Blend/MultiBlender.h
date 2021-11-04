#ifndef MULTI_BLENDER_H
#define MULTI_BLENDER_H

#include "BlendBase.h"
class MultiBlender :
	public BlenderBase
{
	virtual void SoAddBlenderImgs(const cv::Point2f& upper_left, const cv::Point2f& bottom_right, \
		const cv::Mat& img, \
		std::function<cv::Point2f(cv::Point2i)>coor_func) override;

	virtual cv::Mat SoRunBlender(const int& iImgOrder) override;
};
#endif		/*!< MULTI_BLENDER_H */
