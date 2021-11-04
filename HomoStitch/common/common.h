#ifndef	COMMON_H
#define	COMMON_H

#include <opencv2/core.hpp>
#include <vector>
#include <utility>

#include "spdloghelper.h"

using namespace std;

#ifndef IMG_RANGE_T
#define IMG_RANGE_T
typedef struct img_range_t				/*!< 图像投影范围 */
{
	cv::Point2f MinXY;
	cv::Point2f MaxXY;
	img_range_t()
	{
		MinXY.x = MinXY.y = FLT_MAX;
		MaxXY.x = MaxXY.y = FLT_MIN;
	}
}ImgRage_T;

#endif	/*!< IMG_RANGE_T */

#ifndef MATCH_COMPONENT_T
#define MATCH_COMPONENT_T
typedef struct match_component_t		/*!< 图像投影信息 */
{
	cv::Mat HSrcPano;					/*!< Pano->Src */
	ImgRage_T SingleImgRange;
	cv::Mat HPanoSrc;					/*!< Src->Pano */
}MatchComponent_T;
#endif	/*!< MATCH_COMPONENT_T */

#ifndef CONNECTED_IMAGES_T
#define CONNECTED_IMAGES_T
typedef struct connected_images_t		/*!< 图像连接信息 */
{
	ImgRage_T PanoRange;				/*!< 全景图像范围 */
	std::vector<MatchComponent_T> vImagesRane;
}ConnectedImages_T;
#endif	/*!< CONNECTED_IMAGES_T */

typedef struct match_info_t						/*!< 两张图像之间的匹配信息 */
{
	cv::Mat Homo;
	//vector<std::pair<cv::Point2f, cv::Point2f>> vMatchPair;

	vector<cv::Point2f>vKptsI, vKptsJ;			/*!< 特征点 */
	std::pair<cv::Size, cv::Size > MatchShape;

	//vector<int> viMatchIndex;

	vector<std::pair<int, int>> viMatchIndex;	/*!< 匹配点 */

	vector<std::pair<cv::Point2f, cv::Point2f>> vMatchInlierPts;		/*!< 内点(用于) */

	int iInlierSize;							/*!< 内点个数 */

	float fConfidence;

	void reverse()
	{
		//for (auto& c : viMatchIndex)
		//	std::swap(c.first, c.second);
	}
	match_info_t()
	{
		iInlierSize = 0;
		fConfidence = 0;
	}
}MatchInfo_T;

#ifndef PANO_MODEL_M
#define	PANO_MODEL_M
typedef enum pano_model
{
	PANO_FLAT = 0,
	PANO_CYLINDRICAL = 1,
	PANO_SPHERICAL = 2
}PanoModel_M;
#endif

#ifndef MATCH_METHOD_M
#define	MATCH_METHOD_M
typedef enum match_method
{
	MATCH_BF = 0,
	MATCH_FLANN = 1,
	MATCH_GMS = 2
}MatchMethod_M;
#endif

#ifndef FEATURE_METHOD_M
#define	FEATURE_METHOD_M
typedef enum feature_method
{
	FEATURE_SIFT = 0,
	FEATURE_SURF = 1,
	FEATURE_ORB = 2
}FeatureMethod_M;
#endif

#ifndef TRANSFORM_METHOD_M
#define	TRANSFORM_METHOD_M
typedef enum Transform_model
{
	AFFINE = 0,				/*!< 仿射变换2*3 */
	HOMO = 1,				/*!< 透视变化3*3 */
}TransModel_M;
#endif

#ifndef STITCH_CONFIG_M
#define	STITCH_CONFIG_M
typedef struct StitchConfig
{
	int		iPanoModel;
	int		iMatchMethod;
	int		iFeatureMethod;
	int		iOrder;
	int		iBlenderMethod;
	float   fCamFocal;
	TransModel_M	TransformType;
}StitchConfig_T;
#endif

static inline void SoInitLogFile(const string& strLogDir = "PanoLogs")
{
	string strLogFile = strLogDir + "/PanoLog.logs";
	SimLog2::Instance()->InitSimLog("CigarettePano", strLogFile, 1);
}

#endif	/*!< COMMON_H */
