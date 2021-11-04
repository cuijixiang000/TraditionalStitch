#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "StitchBase.h"
#include "StitchFilter.h"
#include "BlendBase.h"
#include "LineBlender.h"
#include "MultiBlender.h"
#include "CylinderWarp.h"

StitchBase::StitchBase(const vector<cv::Mat>& vImg, const StitchConfig& Config)
{
	for (auto const& i : vImg)
	{
		m_vImgs.emplace_back(i);
	}
	m_vvKeyPoints.resize(m_vImgs.size());
	m_vDstps.resize(m_vImgs.size());
	m_Config = Config;
	_iMaxPanoSize = 20000;
}
StitchBase::~StitchBase()
{
	m_vvKeyPoints.clear();
	m_vvKeyPoints.shrink_to_fit();

	m_vDstps.clear();
	m_vDstps.shrink_to_fit();
}

///特征提取+特征描述子
void StitchBase::CalFeatures()
{
	if (FeatureMethod_M::FEATURE_SIFT == m_Config.iFeatureMethod)
	{
		auto sift_detect = cv::xfeatures2d::SIFT::create(500);

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < m_vImgs.size(); i++)
		{
			sift_detect->detectAndCompute(m_vImgs[i], cv::Mat(), m_vvKeyPoints[i], m_vDstps[i]);
			DEBUG("Image {}  has {} features", i, m_vvKeyPoints[i].size());
		}
	}
	else if (FeatureMethod_M::FEATURE_SURF == m_Config.iFeatureMethod)
	{
		auto surf_detect = cv::xfeatures2d::SURF::create(1500);
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < m_vImgs.size(); i++)
		{
			surf_detect->detectAndCompute(m_vImgs[i], cv::Mat(), m_vvKeyPoints[i], m_vDstps[i]);
			DEBUG("Image {}  has {} features", i, m_vvKeyPoints[i].size());
		}
	}
	else if (FeatureMethod_M::FEATURE_ORB == m_Config.iFeatureMethod)
	{
		auto orb_detect = cv::ORB::create(500);
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < m_vImgs.size(); i++)
		{
			orb_detect->detectAndCompute(m_vImgs[i], cv::Mat(), m_vvKeyPoints[i], m_vDstps[i]);
			DEBUG("Image {}  has {} features", i, m_vvKeyPoints[i].size());
		}
	}
}

bool StitchBase::PairWiseMatch(const vector<cv::KeyPoint>& kpI, const cv::Mat& DscptI, const vector<cv::KeyPoint>& kpJ, const cv::Mat& DscptJ, MatchInfo_T& ImgMathInfo)
{
	///暴力匹配
	std::vector<cv::DMatch>GoodMatchPts;
	if (MatchMethod_M::MATCH_BF == m_Config.iMatchMethod)
	{
		BF_Match(DscptI, DscptJ, GoodMatchPts);
	}
	else if (MatchMethod_M::MATCH_FLANN == m_Config.iMatchMethod)
	{
		Flann_Match(DscptI, DscptJ, GoodMatchPts);
	}
	else if (MatchMethod_M::MATCH_GMS == m_Config.iMatchMethod)
	{
		GMS_Match(kpI, DscptI, ImgMathInfo.MatchShape.first, kpJ, DscptJ, ImgMathInfo.MatchShape.second, GoodMatchPts);
	}
	if (GoodMatchPts.size() < 8)
	{
		//cout << "GoodMatchPts num < 8!!!" << endl;
		return false;
	}
	//cv::Mat outImg;
	//cv::drawMatches(m_vImgs[1], m_vvKeyPoints[1], m_vImgs[0], m_vvKeyPoints[0], GoodMatchPts, outImg);
	//cv::imwrite("outImg.png", outImg);
	vector<cv::Point2f>keyptI, keyptJ;
	for (const auto& i : GoodMatchPts)
	{
		//std::pair<cv::Point2f, cv::Point2f> matchPt;
		//matchPt = std::make_pair(kpI[i.trainIdx].pt, kpJ[i.queryIdx].pt);
		//ImgMathInfo.vMatchPair.emplace_back(matchPt);

		keyptI.emplace_back(kpI[i.trainIdx].pt);
		keyptJ.emplace_back(kpJ[i.queryIdx].pt);

		std::pair<int, int> matchID;
		matchID = std::make_pair(i.trainIdx, i.queryIdx);

		ImgMathInfo.viMatchIndex.emplace_back(matchID);
	}

	for (const auto& i : kpI) ImgMathInfo.vKptsI.emplace_back(i.pt);
	for (const auto& i : kpJ) ImgMathInfo.vKptsJ.emplace_back(i.pt);

	///计算H
	cv::Mat mask;

	if (m_Config.TransformType == TransModel_M::HOMO)
	{
		ImgMathInfo.Homo = cv::findHomography(keyptJ, keyptI, mask, cv::RHO, 3.0);
		//ImgMathInfo.Homo = cv::findHomography(keyptJ, keyptI, mask, cv::RANSAC, 3.0);
	}
	else if (m_Config.TransformType == TransModel_M::AFFINE)
	{
		cv::Mat H1 = cv::estimateAffine2D(keyptJ, keyptI, mask, cv::RANSAC, 1.0);
		if (!H1.data)
		{
			return false;
		}
		ImgMathInfo.Homo = cv::Mat::eye(3, 3, CV_64F);
		ImgMathInfo.Homo.at<double>(0, 0) = H1.at<double>(0, 0);
		ImgMathInfo.Homo.at<double>(0, 1) = H1.at<double>(0, 1);
		ImgMathInfo.Homo.at<double>(0, 2) = H1.at<double>(0, 2);

		ImgMathInfo.Homo.at<double>(1, 0) = H1.at<double>(1, 0);
		ImgMathInfo.Homo.at<double>(1, 1) = H1.at<double>(1, 1);
		ImgMathInfo.Homo.at<double>(1, 2) = H1.at<double>(1, 2);
	}

	if (!ImgMathInfo.Homo.data)
	{
		//cout << "Homo is NULL" << endl;
		return false;
	}
	int iInlierNum = cv::sum(mask)[0];
	ImgMathInfo.iInlierSize = iInlierNum;
	if (iInlierNum < 8)
	{
		//cout << "iInlierNum num < 8!!!" << endl;
		return false;
	}
	//ImgMathInfo.iInlierSize = iInlierNum;

	std::vector<cv::DMatch>RansacGoodMatchPts;
	for (int i = 0; i < GoodMatchPts.size(); i++)
	{
		if (mask.at<uchar>(i, 0))		/*!< 挑选的内点 */
		{
			std::pair<cv::Point2f, cv::Point2f>InteriorPoint;
			//std::make_pair(keyptI[GoodMatchPts[i].trainIdx], keyptI[GoodMatchPts[i].queryIdx]);

			InteriorPoint = std::make_pair(keyptI[i], keyptJ[i]);

			ImgMathInfo.vMatchInlierPts.emplace_back(InteriorPoint);

			//RansacGoodMatchPts.push_back(GoodMatchPts[i]);
			//ImgMathInfo.viMatchIndex.emplace_back(i);
		}
	}

	//cv::drawMatches(m_vImgs[1], m_vvKeyPoints[1], m_vImgs[0], m_vvKeyPoints[0], RansacGoodMatchPts, outImg);
	//cv::imwrite("outImg_Ransac.png", outImg);

	return true;
}

bool StitchBase::HFilter(MatchInfo_T& ImgMathInfo)
{
	///难道是因为指针的原因吗？

	std::shared_ptr<StitchFilter>ptrHfilter = std::make_shared<StitchFilter>(ImgMathInfo);
	//return true;

	if (!ptrHfilter->SoStithFliter())
	{
		//cout << "HFilter is error!!!" << endl;
		return false;
	}
	return true;
}

cv::Mat StitchBase::Blender()
{
	bool bCyFlag = false;
	if (m_proj_method == ProjMed::cylindrical)
	{
		m_proj_method = ProjMed::flat;
		bCyFlag = true;
	}
	auto proj2homo = get_proj2homo();

	cv::Point2f resolution = GetFinalResolusion();

	auto pt_proj = [&](const double* ptrH, const cv::Point3f& pt)
	{
		double dX, dY, dZ;
		dX = ptrH[0] * pt.x + ptrH[1] * pt.y + ptrH[2] * pt.z;
		dY = ptrH[3] * pt.x + ptrH[4] * pt.y + ptrH[5] * pt.z;
		dZ = ptrH[6] * pt.x + ptrH[7] * pt.y + ptrH[8] * pt.z;
		cv::Point3f p3d(dX, dY, dZ);
		return p3d;
	};

	///按照分辨率与最小值平移，保证平移后的范围都是正数
	auto scale_coor_to_img_coor = [&](cv::Point2f v)
	{
		v.x = (v.x - m_GroupImgs.PanoRange.MinXY.x) / resolution.x;
		v.y = (v.y - m_GroupImgs.PanoRange.MinXY.y) / resolution.y;
		return cv::Point2f(v.x, v.y);
	};

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

	std::shared_ptr<BlenderBase>classBlender;
	if (m_Config.iBlenderMethod == 0)		/*!< 线性融合 */
	{
		classBlender = std::make_shared<LineBlender>();
	}
	else
	{
		classBlender = std::make_shared<MultiBlender>();
	}
	int iIndex = 0;
	for (const auto i : m_GroupImgs.vImagesRane)
	{
		cv::Point2f top_left = scale_coor_to_img_coor(i.SingleImgRange.MinXY);
		cv::Point2f bottom_right = scale_coor_to_img_coor(i.SingleImgRange.MaxXY);
		cv::Mat Img = m_vImgs[iIndex];

		cv::Point2f CenterXY;

		CenterXY.x = Img.cols / 2;
		CenterXY.y = Img.rows / 2;

		double ptrH[9] = { 0.0 };
		Homo2Ptr(i.HSrcPano, ptrH);

		std::shared_ptr<CylinderWarp> classCy;
		if (bCyFlag)			/*!< 柱面图->原始图 */
		{
			float fFactor = 1.0;
			float fFocal = m_Config.fCamFocal;
			classCy = std::make_shared<CylinderWarp>(fFactor, Img.size(), fFocal);
		}

		classBlender->SoAddBlenderImgs(top_left, bottom_right, Img,
			[=, &Img](cv::Point2i t)->cv::Point2f
		{
			cv::Point2f c;
			c.x = t.x * resolution.x + m_GroupImgs.PanoRange.MinXY.x;
			c.y = t.y * resolution.y + m_GroupImgs.PanoRange.MinXY.y;
			cv::Point3f homo = proj2homo(c);
			cv::Point3f ret = pt_proj(ptrH, homo);
			if (ret.z < 0)
				return cv::Point2f{ -10, -10 };  // was projected to the other side of the lens, discard
			double denom = 1.0 / ret.z;

			cv::Point2f tempdata;
			tempdata.x = float(ret.x * denom);
			tempdata.y = float(ret.y * denom);
			if (bCyFlag)			/*!< 柱面图->原始图 */
			{
				tempdata = classCy->SoPtCylinderProjInv(tempdata);
			}
			return tempdata;
			//return cv::Point2f{ float(ret.x * denom), float(ret.y * denom) };
		});
		iIndex++;
	}

	//cv::Mat ProjImg1, ProjImg2;
	//cv::warpPerspective(m_vImgs[0], ProjImg1, m_GroupImgs.vImagesRane[0].HPanoSrc, cv::Size(2000, 2000));
	//cv::imwrite("Proj1.png", ProjImg1);

	//cv::warpPerspective(m_vImgs[1], ProjImg2, m_GroupImgs.vImagesRane[1].HPanoSrc, cv::Size(2000, 2000));
	//cv::imwrite("Proj2.png", ProjImg2);
	return classBlender->SoRunBlender(m_Config.iOrder);
}

void StitchBase::EdgeSampling(std::vector<cv::Point2f>& vEdgeCorners)
{
	///遍历图像边缘
	const static int CORNER_SAMPLE = 100;          /*!< 以图像中心为原点，间隔100沿着图像边缘采样 */

	for (int i = 0; i < CORNER_SAMPLE; i++)
	{
		double xi = (double)i / CORNER_SAMPLE;
		vEdgeCorners.emplace_back(xi, 0);
		vEdgeCorners.emplace_back(xi, 1);
	}
	for (int j = 0; j < CORNER_SAMPLE; j++)
	{
		double yj = (double)j / CORNER_SAMPLE;
		vEdgeCorners.emplace_back(0, yj);
		vEdgeCorners.emplace_back(1, yj);
	}
}

/*
void StitchBase::UpDateRange()
{
	if (m_Config.iMatchMethod == PanoModel_M::PANO_FLAT
		|| m_Config.iMatchMethod == PanoModel_M::PANO_CYLINDRICAL)
	{
		_proj_method = ProjMed::flat;
	}
	else if (m_Config.iMatchMethod == PanoModel_M::PANO_SPHERICAL)
	{
		_proj_method = ProjMed::spherical;
	}
	auto homo2proj = get_homo2proj();
	auto pt_proj = [&](const double* ptrH, const cv::Point2f& pt)
	{
		double dX, dY, dZ;
		dX = ptrH[0] * pt.x + ptrH[1] * pt.y + ptrH[2] * 1;
		dY = ptrH[3] * pt.x + ptrH[4] * pt.y + ptrH[5] * 1;
		dZ = ptrH[6] * pt.x + ptrH[7] * pt.y + ptrH[8] * 1;
		cv::Point3f p3d(dX, dY, dZ);
		cv::Point2f tempPt;
		tempPt = homo2proj(p3d);
		return tempPt;
	};

	///图像边缘间隔采样
	std::vector<cv::Point2f>vEdgeCorners;
	EdgeSampling(vEdgeCorners);
	auto cal_range = [&](const cv::Mat& Homo, const cv::Size& ImgSize)
	{
		ImgRage_T img_range;
		double ptrH[9] = { 0.0 };
		ptrH[0] = Homo.ptr<double>(0)[0];
		ptrH[1] = Homo.ptr<double>(0)[1];
		ptrH[2] = Homo.ptr<double>(0)[2];

		ptrH[3] = Homo.ptr<double>(1)[0];
		ptrH[4] = Homo.ptr<double>(1)[1];
		ptrH[5] = Homo.ptr<double>(1)[2];

		ptrH[6] = Homo.ptr<double>(2)[0];
		ptrH[7] = Homo.ptr<double>(2)[1];
		ptrH[8] = Homo.ptr<double>(2)[2];

		for (const auto& v : vEdgeCorners)
		{
			cv::Point2f tempdata;
			tempdata.x = v.x * ImgSize.width;
			tempdata.y = v.y * ImgSize.height;
			cv::Point2f t_corner;

			t_corner = pt_proj(ptrH, tempdata);

			img_range.MinXY.x = MIN(img_range.MinXY.x, t_corner.x);
			img_range.MinXY.y = MIN(img_range.MinXY.y, t_corner.y);

			img_range.MaxXY.x = MAX(img_range.MaxXY.x, t_corner.x);
			img_range.MaxXY.y = MAX(img_range.MaxXY.y, t_corner.y);
		}
		return img_range;
	};

	int iIndex = 0;

	for (auto& i : m_GroupImgs.vImagesRane)
	{
		i.SingleImgRange = cal_range(i.HPanoSrc, m_vImgs[iIndex].size());
		iIndex++;
		m_GroupImgs.PanoRange.MaxXY.x = MAX(m_GroupImgs.PanoRange.MaxXY.x, i.SingleImgRange.MaxXY.x);
		m_GroupImgs.PanoRange.MaxXY.y = MAX(m_GroupImgs.PanoRange.MaxXY.y, i.SingleImgRange.MaxXY.y);

		m_GroupImgs.PanoRange.MinXY.x = MIN(m_GroupImgs.PanoRange.MinXY.x, i.SingleImgRange.MinXY.x);
		m_GroupImgs.PanoRange.MinXY.y = MIN(m_GroupImgs.PanoRange.MinXY.y, i.SingleImgRange.MinXY.y);
	}
}
*/

void StitchBase::BF_Match(const cv::Mat& DstpI, const cv::Mat& DstpJ, std::vector<cv::DMatch>& GoodMatchPts)
{
	vector<cv::DMatch>vMatchPoints;
	cv::Ptr<cv::BFMatcher>bf;

	if (FeatureMethod_M::FEATURE_ORB == m_Config.iFeatureMethod)
	{
		bf = cv::BFMatcher::create(cv::NORM_HAMMING, true);
	}
	else
	{
		bf = cv::BFMatcher::create(cv::NORM_L2, true);
	}
	bf->match(DstpJ, DstpI, vMatchPoints);
	double dMinDis = DBL_MAX;
	double dMaxDis = -DBL_MAX;
	for (auto const& i : vMatchPoints)
	{
		dMinDis = MIN(dMinDis, i.distance);
		dMaxDis = MAX(dMaxDis, i.distance);
	}
	for (auto const& i : vMatchPoints)
	{
		if (i.distance < MAX((7.5 * dMinDis), 30))
		{
			GoodMatchPts.emplace_back(i);
		}
	}
}

void StitchBase::Flann_Match(const cv::Mat& DstpI, const cv::Mat& DstpJ, std::vector<cv::DMatch>& GoodMatchPts)
{
	cv::FlannBasedMatcher ANNMatcher;
	std::vector<std::vector<cv::DMatch>>vvMatchPoints;

	cv::Mat tempDstpI, tempDstpJ;
	if ((DstpI.type() != CV_32F || DstpJ.type() != CV_32F))
	{
		DstpI.convertTo(tempDstpI, CV_32F);
		DstpJ.convertTo(tempDstpJ, CV_32F);
	}
	else
	{
		tempDstpI = DstpI;
		tempDstpJ = DstpJ;
	}
	ANNMatcher.knnMatch(tempDstpJ, tempDstpI, vvMatchPoints, 2);

	for (auto const& i : vvMatchPoints)
	{
		if (i[0].distance < 0.6 * i[1].distance)
		{
			GoodMatchPts.emplace_back(i[0]);
		}
	}
}

void StitchBase::GMS_Match(const vector<cv::KeyPoint>& kpI, const cv::Mat& DstpI, const cv::Size& sizeI, const vector<cv::KeyPoint>& kpJ, const cv::Mat& DstpJ, const cv::Size& sizeJ, std::vector<cv::DMatch>& GoodMatchPts)
{
	std::vector<cv::DMatch> matchesAll;

	auto bf = cv::BFMatcher::create(cv::NORM_HAMMING, true);
	bf->match(DstpJ, DstpI, matchesAll);
	cv::xfeatures2d::matchGMS(sizeJ, sizeI, kpJ, kpI, matchesAll, GoodMatchPts, true, true);
}

pano::homo2proj_t StitchBase::get_homo2proj() const
{
	switch (m_proj_method) {
	case ProjMed::flat:
		return pano::flat::homo2proj;
	case ProjMed::cylindrical:
		return pano::cylindrical::homo2proj;
	case ProjMed::spherical:
		return pano::spherical::homo2proj;
	}
	assert(false);
	return pano::flat::homo2proj;
}

pano::proj2homo_t StitchBase::get_proj2homo() const {
	switch (m_proj_method) {
	case ProjMed::flat:
		return pano::flat::proj2homo;
	case ProjMed::cylindrical:
		return pano::cylindrical::proj2homo;
	case ProjMed::spherical:
		return pano::spherical::proj2homo;
	}
	assert(false);
	return pano::flat::proj2homo;
}