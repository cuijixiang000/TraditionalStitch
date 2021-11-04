#include "CylinderWarp.h"

#include "CylinderStitcher.h"

#include <iostream>

cv::Mat CylinderStitcher::SoBuild()
{
	LOG("********Start CylinderStitcher...********");
	CalFeatures();
	if (!CylinderImgsMatch())
	{
		ERROR("CylinderImgsMatch ERROR!!!");
		return cv::Mat();
	}

	///放入m_GroupImgs;
	SelectBaseFrame();
	///计算投影范围
	UpDateRange();
	LOG("********CylinderStitcher is OK********");
	return  Blender();
}

bool CylinderStitcher::CylinderImgsMatch()
{
	bool bSuss = false;
	int iImgsNum = m_vImgs.size();

	_vvMatchsInfo.resize(iImgsNum);
	for (auto& i : _vvMatchsInfo)
	{
		i.resize(iImgsNum);
	}

	///增加柱面点集投影
	float fFactor = 1.0;
	float fFocal = m_Config.fCamFocal;

	_vCySize.resize(m_vImgs.size());

	std::vector<std::vector<cv::KeyPoint>>_vvCyKeyPoints;
	_vvCyKeyPoints.resize(m_vImgs.size());

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < m_vvKeyPoints.size(); i++)
	{
		std::shared_ptr<CylinderWarp> classWarp = std::make_shared<CylinderWarp>(fFactor, m_vImgs[i].size(), fFocal);
		classWarp->SoMatchPtsCylinderWarp(m_vvKeyPoints[i], _vvCyKeyPoints[i]);
		cv::Size tempSize = classWarp->SoGetCylinderSize();
		_vCySize[i] = tempSize;
	}

#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < iImgsNum; i++)
	{
		int iNext = (i + 1) % iImgsNum;

		if (i == iImgsNum - 1)
		{
			continue;
		}

		MatchInfo_T ImgMathInfo;
		ImgMathInfo.MatchShape.first = m_vImgs[i].size();
		ImgMathInfo.MatchShape.second = m_vImgs[iNext].size();

		///增加一个柱面投影

		//bSuss = PairWiseMatch(m_vvKeyPoints[i], m_vDstps[i], m_vvKeyPoints[iNext], m_vDstps[iNext], ImgMathInfo);

		bSuss = PairWiseMatch(_vvCyKeyPoints[i], m_vDstps[i], _vvCyKeyPoints[iNext], m_vDstps[iNext], ImgMathInfo);

		DEBUG("Image {} and {} has {} matched points", i, iNext, ImgMathInfo.iInlierSize);

		if (!bSuss)
		{
			ERROR("Image {} and {} don't match", i, iNext);
			exit(1);
			//return false;
		}

		bSuss = HFilter(ImgMathInfo);

		if (!bSuss)
		{
			ERROR("Image {} and {} HFilter ERROR!!!", i, iNext);
			exit(1);
		}

		//ImgMathInfo.vMatchPair.clear();

		_vvMatchsInfo[i][iNext] = ImgMathInfo;
		//std::cout << iNext << "->" << i << ": " << ImgMathInfo.viMatchIndex.size() << std::endl;
		//DEBUG("{}->{}: {}", i, iNext, ImgMathInfo.iInlierSize);
		///判断是否可逆
		cv::Mat Inv = ImgMathInfo.Homo.inv();
		Inv *= (1.0 / Inv.at<double>(2, 2));
		ImgMathInfo.Homo = Inv;
		ImgMathInfo.reverse();

		_vvMatchsInfo[iNext][i] = move(ImgMathInfo);

		//ImgMathInfo.vKptsI.clear();
		//ImgMathInfo.vKptsJ.clear();
		//ImgMathInfo.viMatchIndex.clear();
	}
	return true;
}

void CylinderStitcher::SelectBaseFrame()
{
	///选择中间帧为基准
  // TODO bfs over pairwise to build bundle
// assume pano pairwise
	int n = m_vImgs.size();
	int mid = m_vImgs.size() >> 1;
	m_GroupImgs.vImagesRane.resize(n);
	m_GroupImgs.vImagesRane[mid].HPanoSrc = cv::Mat::eye(3, 3, CV_64F);

	auto& comp = m_GroupImgs.vImagesRane;

	// accumulate the transformations
	if (mid + 1 < n) {
		comp[mid + 1].HPanoSrc = _vvMatchsInfo[mid][mid + 1].Homo;
		for (int k = mid + 2; k < n; k++)
		{
			comp[k].HPanoSrc = comp[k - 1].HPanoSrc * _vvMatchsInfo[k - 1][k].Homo;
		}
	}
	if (mid - 1 >= 0) {
		comp[mid - 1].HPanoSrc = _vvMatchsInfo[mid][mid - 1].Homo;
		for (int k = mid - 2; k >= 0; k--)
		{
			comp[k].HPanoSrc = comp[k + 1].HPanoSrc * _vvMatchsInfo[k + 1][k].Homo;
		}
	}
	// comp[k]: from k to identity. [-w/2,w/2]

	// when estimate_camera is not used, homo is KRRK(2d-2d), not KR(2d-3d)
	// need to somehow normalize(guess) focal length to make non-flat projection work
	double f = -1;
	f = 0.5 * (m_vImgs[mid].cols + m_vImgs[mid].rows);

	cv::Mat M = cv::Mat::eye(3, 3, CV_64F);
	M.at<double>(0, 0) = 1 / f;
	M.at<double>(1, 1) = 1 / f;

	//	cout << "M: " << M << endl;

	for (int i = 0; i < n; i++)
	{
		comp[i].HPanoSrc = comp[i].HPanoSrc;
		comp[i].HSrcPano = comp[i].HPanoSrc.inv();
		//std::cout << "comp[i].HPanoSrc" << comp[i].HPanoSrc << endl;
	}
	_vvMatchsInfo.clear();
	_vvMatchsInfo.shrink_to_fit();
}

void CylinderStitcher::UpDateRange()
{
	m_proj_method = ProjMed::flat;
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
		//i.SingleImgRange = cal_range(i.HPanoSrc, m_vImgs[iIndex].size());

		i.SingleImgRange = cal_range(i.HPanoSrc, _vCySize[iIndex]);

		iIndex++;
		m_GroupImgs.PanoRange.MaxXY.x = MAX(m_GroupImgs.PanoRange.MaxXY.x, i.SingleImgRange.MaxXY.x);
		m_GroupImgs.PanoRange.MaxXY.y = MAX(m_GroupImgs.PanoRange.MaxXY.y, i.SingleImgRange.MaxXY.y);

		m_GroupImgs.PanoRange.MinXY.x = MIN(m_GroupImgs.PanoRange.MinXY.x, i.SingleImgRange.MinXY.x);
		m_GroupImgs.PanoRange.MinXY.y = MIN(m_GroupImgs.PanoRange.MinXY.y, i.SingleImgRange.MinXY.y);
	}
	m_proj_method = ProjMed::cylindrical;
	_vCySize.clear();
	_vCySize.shrink_to_fit();
}

cv::Point2f CylinderStitcher::GetFinalResolusion()
{
	return cv::Point2f(1, 1);
}