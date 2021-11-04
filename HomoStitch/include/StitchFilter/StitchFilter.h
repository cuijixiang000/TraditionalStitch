#ifndef	STITCHER_FILTER_H
#define STITCHER_FILTER_H

#include "common.h"

class PointInPolygon;

///number of inlier divided by all matches in the overlapping region
#define INLIER_IN_MATCH_RATIO 0.1				/*!< 0.05 */
///number of inlier divided by all keypoints in the overlapping region
#define INLIER_IN_POINTS_RATIO 0.04				/*!< 0.01 */

class PointInPolygon;

class StitchFilter
{
public:

	StitchFilter(MatchInfo_T& ImgMathInfo);
	~StitchFilter();
	bool SoStithFliter();

private:

	bool PoloFilter(const cv::Size& sizeI, const cv::Size& sizeJ, const cv::Mat& Homo, const cv::Mat& HomInv, bool Flag);

	bool Healthy(const cv::Mat Homo);

private:

	//const vector<pair<cv::Point2f, cv::Point2f>>& _vMatchPts;

	const vector<cv::Point2f>& _vKptsI, & _vKptsJ;

	const vector<pair<int, int>>& _ivMatchIndex;
	const cv::Size& _SizeI, _SizeJ;
	const cv::Mat& _Homo;

	const int& _iInlierNums;
	float& _fConfidence;

	float					_fInlierPtsRate;			/*!< 内点/重叠区域匹配点 */
	float					_fOverlapArea;
};
#endif	/*!< STITCHER_FILTER_H */
