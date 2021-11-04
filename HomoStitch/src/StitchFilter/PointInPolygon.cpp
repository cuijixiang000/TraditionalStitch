#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <utility>

#include "PointInPolygon.h"

#include "spdloghelper.h"

namespace OpenStitch
{
	float side(const cv::Point2f& a, const cv::Point2f& b, const cv::Point2f& p) {
		return (b - a).cross(p - a);
	}

	Polygon::Polygon()
	{
	}
	Polygon::~Polygon()
	{
		_PolyCenter = _ProjMin = _ProjMax = cv::Point2f(0, 0);

		_ProjSize.width = _ProjSize.height = 0;
		_vCorners.clear();
		_vCorners.shrink_to_fit();

		_Poly.clear();
		_Poly.shrink_to_fit();

		_PolySlopes.clear();
		_PolySlopes.clear();
	}

	void Polygon::SoOverlapPolygon(const cv::Size& DstSize, const cv::Size& SrcSize, const cv::Mat& Homo, const cv::Mat& HomoInv)
	{
		_ProjMin.x = 0, _ProjMin.y = 0;
		_ProjMax.x = DstSize.width - 1;
		_ProjMax.y = DstSize.height - 1;

		RangeCorners(SrcSize, Homo);
		double ptrH[9] = { 0.0 };
		Mat2Ptr(Homo, ptrH);
		std::vector<cv::Point2f>InImgCorners;
		for (auto& v : _vCorners)
		{
			v.x *= SrcSize.width;
			v.y *= SrcSize.height;

			cv::Point2f t_corner = PtProj(ptrH, v);
			if (InImg(DstSize, t_corner))
			{
				InImgCorners.emplace_back(t_corner);
			}
		}
		_vCorners.clear();
		_vCorners.shrink_to_fit();

		auto DstCorners = GetImgCorners(DstSize);
		Mat2Ptr(HomoInv, ptrH);
		for (const auto& i : DstCorners)
		{
			cv::Point2f t_corner = PtProj(ptrH, i);

			if (InImg(SrcSize, t_corner))
			{
				InImgCorners.emplace_back(i);
			}
		}

		//_Poly = InImgCorners;
		_Poly = convex_hull(InImgCorners);

		CalPolyCenter(_Poly);

		CalPolySlope(_Poly);

		InImgCorners.clear();
		InImgCorners.shrink_to_fit();
	}

	void Polygon::SoGetPoly(std::vector<cv::Point2f>& vPoly)
	{
		for (auto& i : _Poly) vPoly.emplace_back(i);
	}

	cv::Mat Polygon::SoOverlapPolygon_abandon(const cv::Size& DstSize, const cv::Size& SrcSize, const cv::Mat& Homo, const cv::Mat& HomoInv)
	{
		_ProjMin.x = 0, _ProjMin.y = 0;
		_ProjMax.x = DstSize.width - 1;
		_ProjMax.y = DstSize.height - 1;

		cv::Mat OffHomo, OffDstHomo;
		CalProjRange(SrcSize, Homo);
		OffsetHomo(Homo, OffHomo);

		cv::Mat DstImg, DstProjMask;
		DstImg.create(DstSize, CV_8UC1);
		DstImg.setTo(1);
		cv::Mat DstHomo = cv::Mat::eye(3, 3, CV_64F);
		OffsetHomo(DstHomo, OffDstHomo);

		cv::Mat SrcImg, SrcProjMask;
		SrcImg.create(DstSize, CV_8UC1);
		SrcImg.setTo(1);

		//cv::imwrite("SrcImg.png", SrcImg);

		cv::warpPerspective(SrcImg, SrcProjMask, OffHomo, _ProjSize);
		cv::warpPerspective(DstImg, DstProjMask, OffDstHomo, _ProjSize);
		//cv::imwrite("SrcProjMask.png", SrcProjMask);
		//cv::imwrite("DstProjMask.png", DstProjMask);

		_MaskUnion = DstProjMask + SrcProjMask;
		_MaskIntersection = _MaskUnion;

		//cv::threshold(_MaskUnion, _MaskIntersection, 1, 255, cv::THRESH_BINARY);

		//cv::threshold(_MaskUnion, _MaskUnion, 0, 255, cv::THRESH_BINARY);
		return _MaskIntersection;
	}

	float Polygon::SoGetUnionArea_abandon()
	{
		float fArea = cv::sum(_MaskUnion)[0] / 255.0;
		return fArea;
	}

	float Polygon::SoGetIntersectionArea_abandon()
	{
		float fArea = cv::sum(_MaskIntersection)[0] / 255.0;
		return fArea;
	}

	bool Polygon::SoPtInPolygon(const cv::Point2i& p)
	{
		float k = atan2((p.y - _PolyCenter.y), (p.x - _PolyCenter.x));
		auto itr = lower_bound(begin(_PolySlopes), end(_PolySlopes), std::make_pair(k, 0));
		int idx1, idx2;
		if (itr == _PolySlopes.end()) {
			idx1 = _PolySlopes.back().second;
			idx2 = _PolySlopes.front().second;
		}
		else {
			idx2 = itr->second;
			if (itr != _PolySlopes.begin())
				idx1 = (--itr)->second;
			else
				idx1 = _PolySlopes.back().second;
		}
		cv::Point2f p1 = _Poly[idx1], p2 = _Poly[idx2];
		//Vec2D p1 = poly[idx1], p2 = poly[idx2];
		// see if com, p are on the same side to line(p1,p2)
		double o1 = side(p1, p2, _PolyCenter), o2 = side(p1, p2, p);
		if (o1 * o2 < -1e-6)
			return false;
		return true;
	}

	float Polygon::SoGetOverlapArea()
	{
		float fOverlapArea = polygon_area(_Poly);
		return fOverlapArea;
	}
	bool Polygon::SoPtInPolygon_abandon(const cv::Point2i& pt)
	{
		uchar pix = _MaskIntersection.at<uchar>(pt);
		if (pix > 0)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	bool Polygon::CalProjRange(const cv::Size Shape, const cv::Mat& Homo)
	{
		RangeCorners(Shape, Homo);

		double ptrH[9] = { 0.0 };
		Mat2Ptr(Homo, ptrH);

		for (auto& v : _vCorners)
		{
			v.x *= Shape.width;
			v.y *= Shape.height;
			cv::Point2f t_corner;

			t_corner = PtProj(ptrH, v);

			_ProjMin.x = MIN(_ProjMin.x, t_corner.x);
			_ProjMin.y = MIN(_ProjMin.y, t_corner.y);

			_ProjMax.x = MAX(_ProjMax.x, t_corner.x);
			_ProjMax.y = MAX(_ProjMax.y, t_corner.y);
		}
		_ProjSize.width = _ProjMax.x - _ProjMin.x;
		_ProjSize.height = _ProjMax.y - _ProjMin.y;
		return true;
	}

	bool Polygon::OffsetHomo(const cv::Mat& OrgHomo, cv::Mat& OffHomo)
	{
		cv::Mat tempH = cv::Mat::eye(3, 3, CV_64F);
		tempH.at<double>(0, 2) = -_ProjMin.x;
		tempH.at<double>(1, 2) = -_ProjMin.y;
		OffHomo = tempH * OrgHomo;
		return true;
	}

	void Polygon::Mat2Ptr(const cv::Mat H, double* ptrH)
	{
		ptrH[0] = H.ptr<double>(0)[0];
		ptrH[1] = H.ptr<double>(0)[1];
		ptrH[2] = H.ptr<double>(0)[2];

		ptrH[3] = H.ptr<double>(1)[0];
		ptrH[4] = H.ptr<double>(1)[1];
		ptrH[5] = H.ptr<double>(1)[2];

		ptrH[6] = H.ptr<double>(2)[0];
		ptrH[7] = H.ptr<double>(2)[1];
		ptrH[8] = H.ptr<double>(2)[2];
	}

	cv::Point2f Polygon::PtProj(const double* ptrH, const cv::Point2f pt)
	{
		double dX, dY, dZ;
		dX = ptrH[0] * pt.x + ptrH[1] * pt.y + ptrH[2] * 1;
		dY = ptrH[3] * pt.x + ptrH[4] * pt.y + ptrH[5] * 1;
		dZ = ptrH[6] * pt.x + ptrH[7] * pt.y + ptrH[8] * 1;

		cv::Point2f tempPt;
		if (0 == dZ)
		{
			tempPt.x = tempPt.y = 0;
		}
		else
		{
			tempPt.x = dX / dZ;
			tempPt.y = dY / dZ;
		}
		return tempPt;
	}

	float Polygon::SoOverlapRate(const cv::Size& DstSize, const cv::Size& SrcSize, const cv::Mat& Homo)
	{
		_ProjMin.x = 0, _ProjMin.y = 0;
		_ProjMax.x = DstSize.width - 1;
		_ProjMax.y = DstSize.height - 1;

		RangeCorners(SrcSize, Homo);
		double ptrH[9] = { 0.0 };
		Mat2Ptr(Homo, ptrH);

		std::vector<cv::Point2f>InImgCorners;

		for (auto& v : _vCorners)
		{
			v.x *= SrcSize.width;
			v.y *= SrcSize.height;

			cv::Point2f t_corner = PtProj(ptrH, v);

			if (InImg(DstSize, t_corner))
			{
				InImgCorners.emplace_back(t_corner);
			}
		}
		_vCorners.clear();
		_vCorners.shrink_to_fit();

		auto DstCorners = GetImgCorners(DstSize);
		cv::Mat HomoInv = Homo.inv();
		Mat2Ptr(HomoInv, ptrH);
		for (const auto& i : DstCorners)
		{
			cv::Point2f t_corner = PtProj(ptrH, i);

			if (InImg(SrcSize, t_corner))
			{
				InImgCorners.emplace_back(t_corner);
			}
		}
		auto tempPoly = convex_hull(InImgCorners);
		float fOverlapArea = polygon_area(tempPoly);

		float fDstArea = DstSize.width * DstSize.height;
		float fSrcArea = SrcSize.width * SrcSize.height;
		float fMaxArea = MAX(fDstArea, fSrcArea);

		float fRate = fOverlapArea / fMaxArea;

		return fRate;
	}

	void Polygon::RangeCorners(const cv::Size Shape, const cv::Mat& Homo)
	{
		///遍历图像边缘
		const static int CORNER_SAMPLE = 20;          /*!< 以图像中心为原点，间隔100沿着图像边缘采样 */

		for (int i = 0; i < CORNER_SAMPLE; i++)
		{
			double xi = (double)i / CORNER_SAMPLE;
			_vCorners.emplace_back(xi, 0);
			_vCorners.emplace_back(xi, 1);
		}
		for (int j = 0; j < CORNER_SAMPLE; j++)
		{
			double yj = (double)j / CORNER_SAMPLE;
			_vCorners.emplace_back(0, yj);
			_vCorners.emplace_back(1, yj);
		}
	}

	std::vector<cv::Point2f> Polygon::GetImgCorners(const cv::Size& Shape)
	{
		return { cv::Point2f(0,0),cv::Point2f(Shape.width - 1,0) ,cv::Point2f(Shape.width - 1,Shape.height - 1) ,cv::Point2f(0,Shape.height - 1) };
	}

	std::vector<cv::Point2f> Polygon::convex_hull(std::vector<cv::Point2f>& pts)
	{
		if (pts.size() <= 3) return pts;
		//m_assert(pts.size());
		sort(begin(pts), end(pts), [](const cv::Point2f& a, const cv::Point2f& b) {
			if (a.y == b.y)	return a.x < b.x;
			return a.y < b.y;
			});
		std::vector<cv::Point2f> ret;
		ret.emplace_back(pts[0]);
		ret.emplace_back(pts[1]);

		// right link
		int n = pts.size();
		for (int i = 2; i < n; ++i) {
			while (ret.size() >= 2 && side(ret[ret.size() - 2], ret.back(), pts[i]) <= 0)
				ret.pop_back();
			ret.emplace_back(pts[i]);
		}

		// left link
		size_t mid = ret.size();
		ret.emplace_back(pts[n - 2]);
		for (int i = n - 3; i >= 0; --i) {
			while (ret.size() > mid && side(ret[ret.size() - 2], ret.back(), pts[i]) <= 0)
				ret.pop_back();
			ret.emplace_back(pts[i]);
		}
		return ret;
	}

	double Polygon::polygon_area(const std::vector<cv::Point2f>& poly)
	{
		// https://www.wikiwand.com/en/Shoelace_formula
		int n = poly.size();

		double sum = 0;
		for (int i = 0; i < n; ++i) {
			double xi = poly[i].x;
			double yi_next = poly[(i + 1) % n].y;
			double yi_prev = poly[(i + n - 1) % n].y;
			sum += xi * (yi_next - yi_prev);
		}
		return 0.5 * fabs(sum);
	}

	bool Polygon::InImg(const cv::Size& Shape, const cv::Point2f& Pt)
	{
		if (Pt.x < 0 || Pt.x > Shape.width || Pt.y < 0 || Pt.y > Shape.height)
		{
			return false;
		}
		else
		{
			return true;
		}
	}

	void Polygon::CalPolyCenter(std::vector<cv::Point2f>& poly)
	{
		for (const auto& i : poly)
		{
			_PolyCenter += i;
		}
		_PolyCenter *= (1.0 / poly.size());
	}

	void Polygon::CalPolySlope(std::vector<cv::Point2f>& poly)
	{
		int iIndex = 0;
		for (const auto& i : poly)
		{
			float k = atan2((i.y - _PolyCenter.y), (i.x - _PolyCenter.x));
			_PolySlopes.emplace_back(k, iIndex);
			iIndex++;
		}
		sort(_PolySlopes.begin(), _PolySlopes.end());
	}
}