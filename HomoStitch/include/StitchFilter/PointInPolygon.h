/******************************************************************************
 * Copyright (c) 2020, 郑州金惠-机器视觉事业部
 *
 * Project		: CigarettePano
 * Purpose		: 多边形判断
 * Author		: 崔继祥
 * Created		: 2021-06-29
 * Modified by	: Cui Jixiang(崔继祥), cui_20151107@126.com
 * Modified     :
******************************************************************************/
#ifndef POLYGON_H
#define POLYGON_H

#include <opencv2/core.hpp>

namespace OpenStitch {
	class Polygon
	{
	public:

		Polygon();
		~Polygon();
		void SoOverlapPolygon(const cv::Size& DstSize, const cv::Size& SrcSize, const cv::Mat& Homo, const cv::Mat& HomoInv);

		bool SoPtInPolygon(const cv::Point2i& p);

		float SoGetOverlapArea();

		void SoGetPoly(std::vector<cv::Point2f>& vPoly);

		float SoOverlapRate(const cv::Size& DstSize, const cv::Size& SrcSize, const cv::Mat& Homo);

		float SoGetUnionArea_abandon();

		float SoGetIntersectionArea_abandon();

		bool SoPtInPolygon_abandon(const cv::Point2i& pt);

		cv::Mat SoOverlapPolygon_abandon(const cv::Size& DstSize, const cv::Size& SrcSize, const cv::Mat& Homo, const cv::Mat& HomoInv);

	private:

		bool CalProjRange(const cv::Size Shape, const cv::Mat& Homo);

		bool OffsetHomo(const cv::Mat& OrgHomo, cv::Mat& OffHomo);

		void Mat2Ptr(const cv::Mat H, double* ptrH);

		cv::Point2f PtProj(const double* ptrH, const cv::Point2f pt);

		bool InImg(const cv::Size& Shape, const cv::Point2f& Pt);

		///边缘插值
		void RangeCorners(const cv::Size Shape, const cv::Mat& Homo);

		///获取图像四个角坐标
		std::vector<cv::Point2f> GetImgCorners(const cv::Size& Shape);

		///计算凸包
		std::vector<cv::Point2f> convex_hull(std::vector<cv::Point2f>& pts);
		double polygon_area(const std::vector<cv::Point2f>& poly);

		void CalPolyCenter(std::vector<cv::Point2f>& poly);

		void CalPolySlope(std::vector<cv::Point2f>& poly);

	private:

		cv::Mat						_MaskUnion;				/*!< 并集 */

		cv::Mat						_MaskIntersection;		/*!< 交集 */

		cv::Point2f					_ProjMin, _ProjMax;

		//cv::Mat					_OffHomo, _OffHomoInv;

		cv::Size					_ProjSize;

		std::vector<cv::Point2f>	_vCorners;

		std::vector<cv::Point2f>	_Poly;

		cv::Point2f					_PolyCenter;
		std::vector<std::pair<float, int>> _PolySlopes;
	};
}
#endif		/*!< POLYGON_H */
