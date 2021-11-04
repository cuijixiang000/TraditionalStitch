#ifndef LINE_BLENDER_H
#define LINE_BLENDER_H

#include "BlendBase.h"

class LineBlender : public BlenderBase
{
public:

	virtual void SoAddBlenderImgs(const cv::Point2f& upper_left, const cv::Point2f& bottom_right, \
		const cv::Mat& img, \
		std::function<cv::Point2f(cv::Point2i)>coor_func) override;

	virtual cv::Mat SoRunBlender(const int& iImgOrder) override;

private:

	void UpdatePanoSize(const cv::Point2f& bottom_right);

private:

	std::vector<ImageToAdd>			_BlenderImgs;

	cv::Size						_PanoSize;

	cv::Point2f						_MaxXY;
};
#endif		/*!< #endif */
