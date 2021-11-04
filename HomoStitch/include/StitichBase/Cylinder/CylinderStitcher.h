/******************************************************************************
 * Copyright (c) 2020, 郑州金惠-机器视觉事业部
 *
 * Project		: OpenStitch
 * Purpose		: 柱面拼接类
 * Author		: 崔继祥
 * Created		: 2021-07-21
 * Modified by	: Cui Jixiang(崔继祥), cui_20151107@126.com
 * Modified     : 1.完整柱面拼接的测试，焦距的准确度影响较大     2021-07-29
				  2.柱面拼接存在黑边的原因：柱面投影后，图像存在黑边  2021-08-16
******************************************************************************/
#ifndef CYLINDER_STITCHER_H
#define CYLINDER_STITCHER_H

#include "StitchBase.h"
class CylinderStitcher :public StitchBase
{
public:

	CylinderStitcher(const vector<cv::Mat>& vImg, const StitchConfig& Config) : StitchBase(vImg, Config) {}

	virtual cv::Mat SoBuild() override;

private:

	virtual cv::Point2f GetFinalResolusion() override;

	virtual void UpDateRange() override;

	void SelectBaseFrame();

	bool CylinderImgsMatch();

private:

	vector<vector<MatchInfo_T>>		_vvMatchsInfo;

	vector<cv::Size>				_vCySize;			/*!< 柱面图尺寸 */
};
#endif		/*!< CYLINDER_STITCHER_H */
