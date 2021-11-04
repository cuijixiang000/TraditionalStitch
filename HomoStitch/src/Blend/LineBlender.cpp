#include <opencv2/imgproc.hpp>

#include <iostream>

#include "BlendBase.h"
#include "common.h"

#include "LineBlender.h"

void LineBlender::SoAddBlenderImgs(const cv::Point2f& upper_left, const cv::Point2f& bottom_right, \
	const cv::Mat& img, \
	std::function<cv::Point2f(cv::Point2i)>coor_func)
{
	_BlenderImgs.emplace_back(ImageToAdd{ Range{upper_left, bottom_right}, img, coor_func });
	UpdatePanoSize(bottom_right);
}

cv::Mat LineBlender::SoRunBlender(const int& iImgOrder)
{
	_PanoSize.width = _MaxXY.x + 1;
	_PanoSize.height = _MaxXY.y + 1;
	cv::Mat PanoImg;

	cout << "PanoImg: (" << _PanoSize.width << "," << _PanoSize.height << ")" << endl;

	LOG("@SoRunBlender@   PanoImg:({},{})", _PanoSize.width, _PanoSize.height);

	try
	{
		PanoImg = cv::Mat::zeros(_PanoSize, CV_8UC3);;
	}
	catch (const std::exception& e)
	{
		_BlenderImgs.clear();
		_BlenderImgs.shrink_to_fit();
		std::cout << "_PanoSize: " << _PanoSize << endl;
		exit(1);
	}

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < _PanoSize.height; i++)
	{
		uchar* row = PanoImg.ptr(i);
		for (int j = 0; j < _PanoSize.width; j++)
		{
			cv::Point3f isum;
			float wsum = 0;
			for (auto& img : _BlenderImgs)
			{
				if (img.range.contain(i, j))
				{
					cv::Point2f img_coor = img.map_coor(i, j);
					if (img_coor.x < 0 || img_coor.y < 0) continue;
					float r = img_coor.y, c = img_coor.x;
					auto color = interpolate(img.imgref, img_coor);
					if (color.x < 0) continue;
					float	w = 0.5 - fabs(c / img.imgref.cols - 0.5);
					if (!iImgOrder)		/* ·ÇË³ÐòÍ¼Ïñ blend both direction */
						w *= (0.5 - fabs(r / img.imgref.rows - 0.5));
					color *= w;
					isum += color;
					wsum += w;
				}
			}
			if (wsum > 0)	// keep original Color::NO
			{
				cv::Point3f BlenderGray = isum / wsum;
				row[j * 3 + 0] = BlenderGray.x;
				row[j * 3 + 1] = BlenderGray.y;
				row[j * 3 + 2] = BlenderGray.z;
			}
		}
	}
	_BlenderImgs.clear();
	_BlenderImgs.shrink_to_fit();
	return PanoImg;
}

void LineBlender::UpdatePanoSize(const cv::Point2f& bottom_right)
{
	_MaxXY.x = MAX(_MaxXY.x, bottom_right.x);
	_MaxXY.y = MAX(_MaxXY.y, bottom_right.y);
}