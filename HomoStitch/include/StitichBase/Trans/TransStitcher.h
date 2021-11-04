/******************************************************************************
 * Copyright (c) 2020, 郑州金惠-机器视觉事业部
 *
 * Project		: OpenStitch
 * Purpose		: 平移模式图像拼接
 * Author		: 崔继祥
 * Created		: 2021-06-21
 * Modified by	: Cui Jixiang(崔继祥), cui_20151107@126.com
 * Modified     : 1.完成平移模式图像拼接			2021-07-16
 *				  2.增加spdlog 与可调式信息			2021-07-19
******************************************************************************/

#ifndef TRANS_STITCHER_H
#define TRANS_STITCHER_H

#include <vector>
#include <opencv2/core.hpp>

#include "StitchBase.h"

class TransStitcher :public StitchBase
{
public:

	TransStitcher(const vector<cv::Mat>& vImg, const StitchConfig& Config) : StitchBase(vImg, Config) {}
	///构造函数
	virtual cv::Mat SoBuild() override;

private:

	virtual cv::Point2f GetFinalResolusion() override;

	virtual void UpDateRange() override;

	void SelectBaseFrame();

	bool ImgsMatch();

private:

	vector<vector<MatchInfo_T>>		_vvMatchsInfo;
};
#endif		/*!< TRANS_STITCHER_H */
