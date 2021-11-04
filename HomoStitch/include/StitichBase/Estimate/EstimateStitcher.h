#ifndef ESTIMATE_STITCHER_H
#define ESTIMATE_STITCHER_H

///光束法平差模块---首尾重复图像 容易出错

#include "StitchBase.h"

#include <vector>

#include "Camera.h"

class EstimateStitcher : public StitchBase
{
public:
	EstimateStitcher(const vector<cv::Mat>& vImg, const StitchConfig& Config) : StitchBase(vImg, Config) {}

	virtual cv::Mat SoBuild() override;

protected:

	virtual cv::Point2f GetFinalResolusion() override;

private:

	//bool EstimateImgsMatch();

	virtual void UpDateRange() override;

	///优化相机参数
	void Optimize();

	///焦距估计
	bool EstimateFocal();

	void FindfocalsFromHomography(const cv::Mat& H, float& f0, float& f1, bool& bf0, bool& bf1);

	///初始化相机参数
	void SetInitCameraParams();

	///最大生生成树查找中心节点
	void findMaxSpanningTree(int num_images, const vector<vector<MatchInfo_T>>& pairwise_matches, cv::detail::Graph& span_tree, vector<int>& centers);

	///初始化中心节点
	void InitCenterNode(const int& iCenterNode);

	///初始化相机节点参数
	void InitNodePara(const int& iFrom, const int& iTo);

private:

	vector<OpenStitch::Camera>					_vCameras;

	vector<vector<MatchInfo_T>>					_vvMatchsInfo;

	int											_iCenterNode;		/*!< 中心节点ID */

	cv::Point2f									_CenterImgSize;	/*!< 弧度单位 */
};
#endif		/*!< ESTIMATE_STITCHER_H */
