#include "CylinderWarp.h"

CylinderWarp::CylinderWarp(const float& fFactor, const cv::Size& ImgSize, const float& fFocal)
{
	_fFactor = fFactor;
	_ImgSize = ImgSize;
	_CylinderSize = CalCylinderSize(fFocal);
}
CylinderWarp::~CylinderWarp()
{
	_fFactor = 0.0;
	_ImgSize.width = _ImgSize.height = 0;
	_CylinderSize.width = _CylinderSize.height = 0;
}

cv::Size CylinderWarp::SoGetCylinderSize()
{
	return _CylinderSize;
}

///图像柱面投影
cv::Mat	CylinderWarp::SoImgCylinderWarp(const cv::Mat& SrcImg)
{
	cv::Mat CylinderImg = cv::Mat::zeros(_CylinderSize, CV_8UC3);
	for (int i = 0; i < CylinderImg.rows; i++)
	{
		uchar* row = CylinderImg.ptr(i);
		for (int j = 0; j < CylinderImg.cols; j++)
		{
			cv::Point2f Pt, img_coor;
			Pt.x = j;
			Pt.y = i;
			img_coor = SoPtCylinderProjInv(Pt);
			if (img_coor.x < 0 || img_coor.y < 0) continue;
			auto color = interpolate(SrcImg, img_coor);
			if (color.x < 0) continue;
			row[j * 3 + 0] = color.x;
			row[j * 3 + 1] = color.y;
			row[j * 3 + 2] = color.z;
		}
	}
	return CylinderImg;
}

///匹配点柱面投影
void CylinderWarp::SoMatchPtsCylinderWarp(std::vector<cv::KeyPoint>& vSrcPts, std::vector<cv::KeyPoint>& vCylinderPts)
{
	for (const auto& i : vSrcPts)
	{
		cv::KeyPoint Pt;
		Pt.pt = PtCylinderProj(i.pt);
		vCylinderPts.emplace_back(Pt);
	}
}

///单点正投影
cv::Point2f CylinderWarp::PtCylinderProj(const cv::Point2f& Pt)
{
	float fTheta = atan((Pt.x - _ImgSize.width / 2) / _fFocal);
	cv::Point2f CylinderPt;
	CylinderPt.x = _fFocal * fTheta + _CylinderSize.width / 2;
	CylinderPt.y = (Pt.y - _ImgSize.height / 2) / (hypot(Pt.x - _ImgSize.width / 2, _fFocal));
	CylinderPt.y = CylinderPt.y * _fFocal + _CylinderSize.height / 2;
	return CylinderPt;
}

///单点逆投影
cv::Point2f CylinderWarp::SoPtCylinderProjInv(const cv::Point2f& Pt)
{
	cv::Point2f CylinderCenter, ImgCenter;
	CylinderCenter.x = _CylinderSize.width / 2;
	CylinderCenter.y = _CylinderSize.height / 2;

	ImgCenter.x = _ImgSize.width / 2;
	ImgCenter.y = _ImgSize.height / 2;

	cv::Point2f tempPt;
	tempPt = (Pt - CylinderCenter) / _fFocal;

	cv::Point2f SrcPt;

	SrcPt.x = _fFocal * tan(tempPt.x) + ImgCenter.x;
	SrcPt.y = tempPt.y * _fFocal / cos(tempPt.x) + ImgCenter.y;

	if (SrcPt.x < 0 || SrcPt.x >= _ImgSize.width || SrcPt.y < 0 || SrcPt.y >= _ImgSize.height)
		SrcPt.x = SrcPt.y = -999;
	return SrcPt;
}

cv::Size CylinderWarp::CalCylinderSize(const float& fFocal)
{
	cv::Size  CylinderSize;
	_fFocal = hypot(_ImgSize.width, _ImgSize.height) * (fFocal / 43.266);
	CylinderSize.width = 2 * _fFocal * atan(_ImgSize.width / (2 * _fFocal));
	CylinderSize.height = _ImgSize.height;
	return CylinderSize;
}

cv::Point3f CylinderWarp::interpolate(const cv::Mat& Img, const cv::Point2f& Pano2ImgUV)
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