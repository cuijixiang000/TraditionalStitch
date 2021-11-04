#ifndef STITCH_BASE_H
#define STITCH_BASE_H

#include <memory>

#include "common.h"
#include "Projection.h"

#ifndef PROJECTIOON_METHOD_M
#define PROJECTIOON_METHOD_M
typedef enum projection_method_m
{
	flat, cylindrical, spherical
}ProjMed;
#endif

class StitchBase
{
public:

	StitchBase(const vector<cv::Mat>& vImg, const StitchConfig& Config);
	~StitchBase();

	virtual cv::Mat SoBuild() = 0;

protected:

	///特征提取+特征描述子
	void CalFeatures();

	bool PairWiseMatch(const vector<cv::KeyPoint>& kpI, const cv::Mat& DscptI, const vector<cv::KeyPoint>& kpJ, const cv::Mat& DscptJ, MatchInfo_T& ImgMathInfo);

	bool HFilter(MatchInfo_T& ImgMathInfo);

	cv::Mat Blender();

	virtual void UpDateRange() = 0;

	//void UpDateRange();

	///边缘采样
	void EdgeSampling(std::vector<cv::Point2f>& vEdgeCorners);

	pano::homo2proj_t get_homo2proj() const;

	pano::proj2homo_t get_proj2homo() const;

	virtual cv::Point2f GetFinalResolusion() = 0;

protected:

	ProjMed			m_proj_method;
private:

	//cv::Point2f GetFinalResolusion();

	void BF_Match(const cv::Mat& DstpI, const cv::Mat& DstpJ, std::vector<cv::DMatch>& GoodMatchPts);

	void Flann_Match(const cv::Mat& DstpLeft, const cv::Mat& DstpRight, std::vector<cv::DMatch>& GoodMatchPts);

	void GMS_Match(const vector<cv::KeyPoint>& kpI, const cv::Mat& DstpI, const cv::Size& sizeI, const vector<cv::KeyPoint>& kpJ, const cv::Mat& DstpJ, const cv::Size& sizeJ, std::vector<cv::DMatch>& GoodMatchPts);

protected:

	std::vector<cv::Mat>					m_vImgs;

	std::vector<std::vector<cv::KeyPoint>>	m_vvKeyPoints;
	std::vector<cv::Mat>					m_vDstps;

	ConnectedImages_T						m_GroupImgs;

	StitchConfig							m_Config;

	int										_iMaxPanoSize;

private:

	//StitchConfig _Config;
};
#endif		/*!< STITCH_BASE_H */
