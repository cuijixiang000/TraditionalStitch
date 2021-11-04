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
	int iFrom;								/*!< �ߵ���ʼ�� */
	int iTo;								/*!< �ߵ��յ� */
	const MatchInfo_T& EdgeMatchInfo;		/*!< �߶�Ӧ��ƥ����Ϣ */
	connect_edge_t(const int& from, const int& to, const MatchInfo_T& EdgeMatchInfo) :
		iFrom(from), iTo(to), EdgeMatchInfo(EdgeMatchInfo) {}
}ConnectEdge_T;
#endif		/*!< CONNECT_EDGE_T */

///L_M����
#define LM_PARA_TAU		1e-3
#define LM_PARA_THETA1	1e-15
#define LM_PARA_THETA2	1e-15
#define LM_PARA_THETA3	1e-15
#define LM_PARA_KMAX	1e+2

#define NR_PARAM_PER_CMAERA 6				/*!< ������������� */
#define	NR_TERM_PER_MATCH 2					/*!< һ��ƥ���Ӧ�������� */

class IncrementalBundleAdjuster
{
public:

	IncrementalBundleAdjuster(std::vector<OpenStitch::Camera>& vCamers);

	///LM�Ż�
	void SoLMOptimize();

	///��˹ţ���Ż�
	void SoGNOptimize();

	///���ӱ�
	void SoAddEdge(const int& iFrom, const int& iTo, const MatchInfo_T& MatchEdge);

	void SoSetCenterID(const int& iCenterNode);

private:

	///�����ſɱ�
	void calcJacobian(Eigen::MatrixXd& J, Eigen::MatrixXd& JTJ);

	///�������
	void calcError(const Eigen::VectorXd& CamPara, Eigen::VectorXd& error);

	///����LM��������
	void CalU(const Eigen::MatrixXd& JTJ);
	double CalRho(const Eigen::VectorXd& CamNew);

	///��ʼ�� _vIndexMap_Org2Add
	void UpdateIndexMap();

	///��������
	void UpdateLMMat();

	///��ʼ���������
	void InitCamaPara(const std::vector<OpenStitch::Camera>& vCamers);
	void UpDateCamPara(std::vector<OpenStitch::Camera>& vCamers);

	///�����������
	void ObtainRefinedCameraParams(std::vector<OpenStitch::Camera>& vCamers);

	///	���� dR/dso3
	std::array<Eigen::Matrix3d, 3> dR_dso3(const cv::Mat& R);

private:

	std::vector<OpenStitch::Camera>& _ResultCam;

	std::vector<ConnectEdge_T>				_vMatchEdges;			/*!< ����ƥ���ϵ�ı� */

	std::vector<int>						_vEdgesPts;				/*!< ÿ���ߵ�ƥ��� */

	std::set<int>							_vOrgImgsID;			/*!< ԭʼͼ��ID,ͬ��Ҳ�ǱߵĶ˵� */

	std::vector<int>						_vIndexMap_Org2Add;		/*!< ԭʼͼ��->����ӵ�BA�еıߵĶ˵�ӳ�� */

	int										_iTotalPtsSum;			/*!< ƥ����ܸ��� */
	int										_iCameraSum;

	///LM����
	int										_iV;

	double									_dU, _dRho, _dV;		/*!< LM�������� */

	Eigen::MatrixXd							_J, _JTJ;

	Eigen::VectorXd							_Error, _G;				/*< ����������� G = -JT*_Error*/

	Eigen::VectorXd							_CamIncrement;			/*< ����������� */
	Eigen::VectorXd							_CamParaIteration;		/*!< ��������������� */

	bool									_bStop;

	int										_iFixedNode;			/*!< �̶����Ľڵ� */
};
#endif		/*!< INCREMENTAL_BUNDLE_ADJUSTER_H */
