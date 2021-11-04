#ifndef INCREMENTAL_BUNDLE_ADJUSTER_H
#define	INCREMENTAL_BUNDLE_ADJUSTER_H

#include <vector>
#include <set>
#include <array>

#include <Eigen/Core>

#include "Camera.h"

#include "common.h"

#ifndef CONNECT_EDGE_T
#define CONNECT_EDGE_T
typedef struct connect_edge_t
{
	int iFrom;								/*!< 边的起始点 */
	int iTo;								/*!< 边的终点 */
	const MatchInfo_T& EdgeMatchInfo;		/*!< 边对应的匹配信息 */
	connect_edge_t(const int& from, const int& to, const MatchInfo_T& EdgeMatchInfo) :
		iFrom(from), iTo(to), EdgeMatchInfo(EdgeMatchInfo) {}
}ConnectEdge_T;
#endif		/*!< CONNECT_EDGE_T */

///L_M参数
#define LM_PARA_TAU		1e-3
#define LM_PARA_THETA1	1e-15
#define LM_PARA_THETA2	1e-15
#define LM_PARA_THETA3	1e-15
#define LM_PARA_KMAX	1e+2

#define NR_PARAM_PER_CMAERA 6				/*!< 单相机参数个数 */
#define	NR_TERM_PER_MATCH 2					/*!< 一次匹配对应的误差个数 */

class IncrementalBundleAdjuster
{
public:

	IncrementalBundleAdjuster(std::vector<OpenStitch::Camera>& vCamers);

	///LM优化
	void SoLMOptimize();

	///高斯牛顿优化
	void SoGNOptimize();

	///增加边
	void SoAddEdge(const int& iFrom, const int& iTo, const MatchInfo_T& MatchEdge);

	void SoSetCenterID(const int& iCenterNode);

private:

	///计算雅可比
	void calcJacobian(Eigen::MatrixXd& J, Eigen::MatrixXd& JTJ);

	///计算误差
	void calcError(const Eigen::VectorXd& CamPara, Eigen::VectorXd& error);

	///计算LM迭代参数
	void CalU(const Eigen::MatrixXd& JTJ);
	double CalRho(const Eigen::VectorXd& CamNew);

	///初始化 _vIndexMap_Org2Add
	void UpdateIndexMap();

	///计算误差方程
	void UpdateLMMat();

	///初始化相机参数
	void InitCamaPara(const std::vector<OpenStitch::Camera>& vCamers);
	void UpDateCamPara(std::vector<OpenStitch::Camera>& vCamers);

	///更新相机参数
	void ObtainRefinedCameraParams(std::vector<OpenStitch::Camera>& vCamers);

	///	求导数 dR/dso3
	std::array<Eigen::Matrix3d, 3> dR_dso3(const cv::Mat& R);

private:

	std::vector<OpenStitch::Camera>& _ResultCam;

	std::vector<ConnectEdge_T>				_vMatchEdges;			/*!< 存在匹配关系的边 */

	std::vector<int>						_vEdgesPts;				/*!< 每条边的匹配点 */

	std::set<int>							_vOrgImgsID;			/*!< 原始图像ID,同样也是边的端点 */

	std::vector<int>						_vIndexMap_Org2Add;		/*!< 原始图像->被添加到BA中的边的端点映射 */

	int										_iTotalPtsSum;			/*!< 匹配点总个数 */
	int										_iCameraSum;

	///LM参数
	int										_iV;

	double									_dU, _dRho, _dV;		/*!< LM迭代参数 */

	Eigen::MatrixXd							_J, _JTJ;

	Eigen::VectorXd							_Error, _G;				/*< 相机参数增量 G = -JT*_Error*/

	Eigen::VectorXd							_CamIncrement;			/*< 相机参数增量 */
	Eigen::VectorXd							_CamParaIteration;		/*!< 迭代过中相机参数 */

	bool									_bStop;

	int										_iFixedNode;			/*!< 固定中心节点 */
};
#endif		/*!< INCREMENTAL_BUNDLE_ADJUSTER_H */
