#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "StitchFilter.h"

#include "PointInPolygon.h"

StitchFilter::StitchFilter(MatchInfo_T& ImgMathInfo) :_vKptsI(ImgMathInfo.vKptsI), _vKptsJ(ImgMathInfo.vKptsJ), _ivMatchIndex(ImgMathInfo.viMatchIndex)\
, _Homo(ImgMathInfo.Homo), _SizeI(ImgMathInfo.MatchShape.first), _SizeJ(ImgMathInfo.MatchShape.second), _iInlierNums(ImgMathInfo.iInlierSize), _fConfidence(ImgMathInfo.fConfidence)
{
	_fInlierPtsRate = 0.0;
	_fOverlapArea = 0.0;
}
StitchFilter::~StitchFilter()
{
}

bool StitchFilter::SoStithFliter()
{
	if (!Healthy(_Homo))
	{
		ERROR("Healthy ERROR!!!");
		return false;
	}
	cv::Mat _HomoInv = _Homo.inv();
	if (0 == cv::countNonZero(_HomoInv))
	{
		return false;
	}
	///判断多边形J->I
	if (!PoloFilter(_SizeI, _SizeJ, _Homo, _HomoInv, true))
	{
		ERROR("PoloFilter ERROR!!!");
		return false;
	}

	float r1p = _fInlierPtsRate;
	float fOverlapArea1 = _fOverlapArea;
	///判断多边形J->I

	//if (!PoloFilter(_SizeJ, _SizeI, _HomoInv, _Homo, false))
	if (!PoloFilter(_SizeI, _SizeJ, _Homo, _HomoInv, false))
	{
		ERROR("PoloFilter ERROR!!!");
		return false;
	}
	float r2p = _fInlierPtsRate;
	float fOverlapArea2 = _fOverlapArea;

	float fConfidence = (r1p + r2p) * 0.5;
	if (fConfidence < INLIER_IN_POINTS_RATIO)
	{
		ERROR("PoloFilter fConfidence({}) < {}!!!", fConfidence, INLIER_IN_POINTS_RATIO);
		return false;
	}
	_fConfidence = fConfidence;
	int area1 = (_SizeI.width * _SizeI.height);
	int area2 = (_SizeJ.width * _SizeJ.height);
	float fAreaRate = max(fOverlapArea1, fOverlapArea2) / max(area1, area2);

	DEBUG("fOverlapArea / area = {}", fAreaRate);
	if (fAreaRate < 0.15)			/*!< 重叠率<0.15 */
	{
		ERROR("PoloFilter fOverlapArea/area({}) < 0.1!!!", fAreaRate);
		return false;
	}
	return true;
}

bool StitchFilter::PoloFilter(const cv::Size& sizeI, const cv::Size& sizeJ, const cv::Mat& Homo, const cv::Mat& HomInv, bool Flag)
{
	std::shared_ptr<OpenStitch::Polygon> ptrPoly = std::make_shared<OpenStitch::Polygon>();
	cv::Mat PolyImg = cv::Mat::zeros(sizeI, CV_8UC3);
	if (Flag)
	{
		ptrPoly->SoOverlapPolygon(sizeI, sizeJ, Homo, HomInv);
		/*
				vector<vector<cv::Point2f>> vPoly(1);
				ptrPoly->SoGetPoly(vPoly[0]);

				//cv::Mat PolyImg = cv::Mat::zeros(sizeI, CV_8UC3);
				for (auto i : vPoly[0])
				{
					cv::circle(PolyImg, i, 10, cv::Scalar(255, 255, 255));
				}
				cv::imwrite("PolyImg0.png", PolyImg);
		*/
	}
	else
	{
		ptrPoly->SoOverlapPolygon(sizeJ, sizeI, HomInv, Homo);
		/*
				vector<cv::Point2f> vPoly;
				ptrPoly->SoGetPoly(vPoly);
				//cv::Mat PolyImg = cv::Mat::zeros(sizeI, CV_8UC1);
				for (auto i : vPoly)
				{
					cv::circle(PolyImg, i, 10, cv::Scalar(255, 255, 255));
				}
				cv::imwrite("PolyImg1.png", PolyImg);
		*/
	}

	///计算重叠区域的匹配点
	int iInlerInOverlap = 0;
	for (const auto& p : _ivMatchIndex)
	{
		if (ptrPoly->SoPtInPolygon(Flag ? _vKptsI[p.first] : _vKptsJ[p.second]))
			iInlerInOverlap++;
	}

	///计算重叠区域的关键点
	int iMatchPtInOverlap = 0;
	for (const auto& p : Flag ? _vKptsI : _vKptsJ)
	{
		if (ptrPoly->SoPtInPolygon(p))
		{
			iMatchPtInOverlap++;
			//cv::circle(PolyImg, p, 5, cv::Scalar(0, 0, 255));
			continue;
		}
		//cv::circle(PolyImg, p, 5, cv::Scalar(255));
	}
	//cv::imwrite("PolyImgInlier.png", PolyImg);

	///内点/重叠区域内的匹配点
	float r1m = _iInlierNums * 1.0f / iInlerInOverlap;
	if (r1m < INLIER_IN_MATCH_RATIO)
	{
		ERROR("_iInlierNums/iInlerInOverlap = {} < {}", r1m, INLIER_IN_MATCH_RATIO);
		return false;
	}
	///内点/重叠区域内的特征点
	_fInlierPtsRate = _iInlierNums * 1.0f / iMatchPtInOverlap;
	if (_fInlierPtsRate < 0.01 || _fInlierPtsRate > 1.0)
	{
		ERROR("_iInlierNums/iMatchPtInOverlap = {} {}~{}", _fInlierPtsRate, 0.01, 1.0);
		return false;
	}
	DEBUG("_iInlierNums/MatchPtsInOverlap = {},_iInlierNums/KeyPtsInOverlap = {}", r1m, _fInlierPtsRate);
	_fOverlapArea = ptrPoly->SoGetOverlapArea();
	return true;
}

bool StitchFilter::Healthy(const cv::Mat Homo)
{
	double ptrH[9] = { 0.0 };
	auto Homo2Ptr = [](const cv::Mat& Homo, double* ptrH)
	{
		ptrH[0] = Homo.ptr<double>(0)[0];
		ptrH[1] = Homo.ptr<double>(0)[1];
		ptrH[2] = Homo.ptr<double>(0)[2];

		ptrH[3] = Homo.ptr<double>(1)[0];
		ptrH[4] = Homo.ptr<double>(1)[1];
		ptrH[5] = Homo.ptr<double>(1)[2];

		ptrH[6] = Homo.ptr<double>(2)[0];
		ptrH[7] = Homo.ptr<double>(2)[1];
		ptrH[8] = Homo.ptr<double>(2)[2];
	};

	Homo2Ptr(Homo, ptrH);
	float HOMO_MAX_PERSPECTIVE = 2e-3;
	// perspective test
	if (fabs(ptrH[6]) > HOMO_MAX_PERSPECTIVE)
		return false;
	if (fabs(ptrH[7]) > HOMO_MAX_PERSPECTIVE)
		return false;
	// flip test
	cv::Vec3d x0(ptrH[2], ptrH[5], ptrH[8]);
	cv::Vec3d x1(ptrH[1] + ptrH[2], ptrH[4] + ptrH[5], ptrH[7] + ptrH[8]);

	if (x1[1] <= x0[1])
		return false;

	cv::Vec3d x2(ptrH[0] + ptrH[1] + ptrH[2],
		ptrH[3] + ptrH[4] + ptrH[5],
		ptrH[6] + ptrH[7] + ptrH[8]);
	if (x2[0] <= x1[0])
		return false;
	return true;
}