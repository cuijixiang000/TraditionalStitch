#include "BlendBase.h"

cv::Point3f BlenderBase::interpolate(const cv::Mat& Img, const cv::Point2f& Pano2ImgUV)
{
	uchar* dataDst = Img.data;
	int stepDst = Img.step; //宽*3

	float fy = Pano2ImgUV.y;

	int sy = cvFloor(fy);
	fy -= sy;
	if (sy < 0)	fy = sy = 0;
	if (sy >= Img.rows - 1)
	{
		fy = 1;
		sy = Img.rows - 2;
	}
	short cbufy[2];
	cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);
	cbufy[1] = 2048 - cbufy[0];

	float fx = Pano2ImgUV.x;
	int sx = cvFloor(fx);
	fx -= sx;

	if (sx < 0) {
		fx = 0, sx = 0;
	}
	if (sx >= Img.cols - 1) {
		fx = 1, sx = Img.cols - 2;
	}
	short cbufx[2];
	cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);
	cbufx[1] = 2048 - cbufx[0];

	float ptrGray[3] = { 0.0 };

	for (int k = 0; k < Img.channels(); ++k)
	{
		ptrGray[k] = (*(dataDst + sy * stepDst + 3 * sx + k) * cbufx[0] * cbufy[0] +
			*(dataDst + (sy + 1) * stepDst + 3 * sx + k) * cbufx[0] * cbufy[1] +
			*(dataDst + sy * stepDst + 3 * (sx + 1) + k) * cbufx[1] * cbufy[0] +
			*(dataDst + (sy + 1) * stepDst + 3 * (sx + 1) + k) * cbufx[1] * cbufy[1]) >> 22;
	}
	return cv::Point3f(ptrGray[0], ptrGray[1], ptrGray[2]);
}