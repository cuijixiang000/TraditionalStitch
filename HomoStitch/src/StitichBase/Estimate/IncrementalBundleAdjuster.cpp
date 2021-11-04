#include <Eigen/Dense>

#include <cmath>
#include <iostream>

#include "IncrementalBundleAdjuster.h"

//#define  DK_DFOCAL Eigen::Matrix3d{1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,0.0}	/*!< dK/df */
//#define  DK_DCX Eigen::Matrix3d{0.0,0.0,1.0, 0.0,0.0,0.0, 0.0,0.0,0.0}		/*!< dK/dcx */
//#define  DK_DCY Eigen::Matrix3d{0.0,0.0,O.0, 0.0,0.0,1.0, 0.0,0.0,0.0}		/*!< dK/dcy */

///反对称矩阵
inline Eigen::Matrix3d SkewSymMetric(const Eigen::Vector3d vct)
{
	double x, y, z;
	x = vct[0];
	y = vct[1];
	z = vct[2];

	Eigen::Matrix3d data;
	data << 0.0, -z, y, z, 0.0, -x, -y, x, 0.0;

	return data;
}

inline Eigen::Matrix3d DK_DFOCAL()
{
	Eigen::Matrix3d data;
	data << 1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 0.0;
	return data;
}
inline Eigen::Matrix3d DK_DCX()
{
	Eigen::Matrix3d data;
	data << 0.0, 0.0, 1.0,
		0.0, 0.0, 0.0,
		0.0, 0.0, 0.0;
	return data;
}
inline Eigen::Matrix3d DK_DCY()
{
	Eigen::Matrix3d data;
	data << 0.0, 0.0, 0.0,
		0.0, 0.0, 1.0,
		0.0, 0.0, 0.0;
	return data;
}

IncrementalBundleAdjuster::IncrementalBundleAdjuster(std::vector<OpenStitch::Camera>& vCamers) :_ResultCam(vCamers), _vIndexMap_Org2Add(vCamers.size())
{
	_iV = 2;
	_iTotalPtsSum = 0;
	_bStop = false;
}

void IncrementalBundleAdjuster::SoGNOptimize()
{
	//return;
	_iCameraSum = _vOrgImgsID.size();
	UpdateIndexMap();
	///雅可比矩阵  行向量偏导
	_J = Eigen::MatrixXd{ NR_TERM_PER_MATCH * _iTotalPtsSum ,NR_PARAM_PER_CMAERA * _iCameraSum };
	_JTJ = Eigen::MatrixXd{ NR_PARAM_PER_CMAERA * _iCameraSum ,NR_PARAM_PER_CMAERA * _iCameraSum };
	_Error = Eigen::VectorXd{ NR_TERM_PER_MATCH * _iTotalPtsSum };
	_G = Eigen::VectorXd{ NR_PARAM_PER_CMAERA * _iCameraSum };

	InitCamaPara(_ResultCam);			/*!< 初始化_CamParaIteration */

	int iK = 0;
	_bStop = false;

	float fInitErr = _Error.lpNorm<1>() / _iTotalPtsSum;
	while (!_bStop && iK < LM_PARA_KMAX)
	{
		++iK;
		UpdateLMMat();
		LOG("BA-LM: {}: Max err: {:.2f},Average err: {:.2f}", iK, _Error.maxCoeff(), _Error.lpNorm<1>() / _iTotalPtsSum);
		_CamIncrement = _JTJ.colPivHouseholderQr().solve(_G).eval();

		if (_CamIncrement.norm() < LM_PARA_THETA2 * _CamParaIteration.norm())
		{
			LOG("||_CamIncrement||: {},||_CamParaIteration||: {}", _CamIncrement.norm(), LM_PARA_THETA2 * _CamParaIteration.norm());

			_bStop = true;
		}
		else
		{
			Eigen::VectorXd CamPara_new = _CamParaIteration + _CamIncrement;

			Eigen::VectorXd ErrorNew = Eigen::VectorXd::Zero(NR_TERM_PER_MATCH * _iTotalPtsSum);
			calcError(CamPara_new, ErrorNew);

			if (ErrorNew.squaredNorm() > _Error.squaredNorm())
			{
				_bStop = true;
				break;
			}
			_CamParaIteration = CamPara_new;
			UpDateCamPara(_ResultCam);
		} while (_dRho < 0 && !_bStop);
	}

	float fFinaleErr = _Error.lpNorm<1>() / _iTotalPtsSum;

	cout << "BA-LM: " << fInitErr << " ----> " << fFinaleErr << endl;
	ObtainRefinedCameraParams(_ResultCam);

	OpenStitch::Camera::SoWaveStraighten(_ResultCam);
}

void IncrementalBundleAdjuster::SoLMOptimize()
{
	//return;
	_iCameraSum = _vOrgImgsID.size();
	UpdateIndexMap();
	///雅可比矩阵  行向量偏导
	_J = Eigen::MatrixXd{ NR_TERM_PER_MATCH * _iTotalPtsSum ,NR_PARAM_PER_CMAERA * _iCameraSum };
	_JTJ = Eigen::MatrixXd{ NR_PARAM_PER_CMAERA * _iCameraSum ,NR_PARAM_PER_CMAERA * _iCameraSum };
	_Error = Eigen::VectorXd{ NR_TERM_PER_MATCH * _iTotalPtsSum };
	_G = Eigen::VectorXd{ NR_PARAM_PER_CMAERA * _iCameraSum };

	InitCamaPara(_ResultCam);			/*!< 初始化_CamParaIteration */
	UpdateLMMat();

	if (_G.lpNorm<Eigen::Infinity>() < LM_PARA_THETA1)
	{
		///直接返回解
	}
	CalU(_JTJ);
	int iK = 0;
	_bStop = false;
	_dRho = 0;
	_dV = 2;

	float fInitErr = _Error.lpNorm<1>() / _iTotalPtsSum;
	while (!_bStop && iK < LM_PARA_KMAX)
	{
		++iK;
		LOG("BA-LM: {}: Max err: {:.2f},Average err: {:.2f}  dRho:{}  _dU:{}", iK, _Error.maxCoeff(), _Error.lpNorm<1>() / _iTotalPtsSum, _dRho, _dU);
		do
		{
			///
			Eigen::MatrixXd U_I = Eigen::MatrixXd::Identity(_JTJ.rows(), _JTJ.cols());
			//_JTJ = _JTJ + _dU * U_I;
			_JTJ = _JTJ + _dU * U_I;
			_CamIncrement = _JTJ.colPivHouseholderQr().solve(_G).eval();
			//if (_CamIncrement.squaredNorm() < LM_PARA_THETA2 * _CamParaIteration.squaredNorm())
			if (_CamIncrement.norm() < LM_PARA_THETA2 * _CamParaIteration.norm())
			{
				LOG("||_CamIncrement||: {},||_CamParaIteration||: {}", _CamIncrement.norm(), LM_PARA_THETA2 * _CamParaIteration.norm());

				_bStop = true;
			}
			else
			{
				//Eigen::VectorXd CamPara_new = Eigen::VectorXd{ NR_PARAM_PER_CMAERA * _iCameraSum };

				//CamPara_new.setZero();
				////Eigen::VectorXd CamPara_new  = _CamParaIteration + _CamIncrement;
				//for (int i = 0; i < NR_PARAM_PER_CMAERA * _iCameraSum; i++)
				//{
				//	if (i < _iFixedNode * NR_PARAM_PER_CMAERA + 3 || i >= _iFixedNode * NR_PARAM_PER_CMAERA + NR_PARAM_PER_CMAERA)
				//	{
				//		CamPara_new(i) = _CamParaIteration(i) + _CamIncrement(i);
				//	}
				//	else
				//	{
				//		CamPara_new(i) = _CamParaIteration(i);
				//	}
				//}
				Eigen::VectorXd CamPara_new = _CamParaIteration + _CamIncrement;

				_dRho = CalRho(CamPara_new);
				if (_dRho > 0)
				{
					_CamParaIteration = CamPara_new;
					UpDateCamPara(_ResultCam);
					UpdateLMMat();
					//	cout << "_G.lpNorm<Eigen::Infinity>(): " << _G.lpNorm<Eigen::Infinity>() << endl;
					//if (_G.lpNorm<Eigen::Infinity>() <= LM_PARA_THETA1 || pow(_Error.squaredNorm(), 2) <= LM_PARA_THETA3)
					if (_G.lpNorm<Eigen::Infinity>() <= LM_PARA_THETA1 || _Error.squaredNorm() <= LM_PARA_THETA3)
					{
						_bStop = true;
					}
					double dTempData = 1 - pow((2 * _dRho - 1), 3);
					_dU = _dU * MAX(1 / 3.0, dTempData);
					_dV = 2;
				}
				else
				{
					_dU *= _dV; _dV *= 2;
				}
			}
		} while (_dRho < 0 && !_bStop);
	}

	float fFinaleErr = _Error.lpNorm<1>() / _iTotalPtsSum;

	cout << "BA-LM: " << fInitErr << " ----> " << fFinaleErr << endl;
	ObtainRefinedCameraParams(_ResultCam);
	//for (auto i : _ResultCam)
	//{
	//	cout << "R: " << i.mR << endl;
	//	cout << "K: " << i.mK << endl;
	//}

	OpenStitch::Camera::SoWaveStraighten(_ResultCam);

	//for (auto i : _ResultCam)
	//{
	//	cout << "Wave_R: " << i.mR << endl;
	//	cout << "i: " << i.mK << endl;
	//}
}

///增加用于平差的边
void IncrementalBundleAdjuster::SoAddEdge(const int& iFrom, const int& iTo, const MatchInfo_T& MatchEdge)
{
	_vMatchEdges.emplace_back(iFrom, iTo, MatchEdge);
	_vEdgesPts.emplace_back(_iTotalPtsSum);
	_iTotalPtsSum += MatchEdge.iInlierSize;
	_vOrgImgsID.insert(iFrom);
	_vOrgImgsID.insert(iTo);
}

void IncrementalBundleAdjuster::SoSetCenterID(const int& iCenterNode)
{
	_iFixedNode = iCenterNode;
}

///计算雅可比
void IncrementalBundleAdjuster::calcJacobian(Eigen::MatrixXd& J, Eigen::MatrixXd& JTJ)
{
	///雅可比矩阵的求解写错了   2021-10-27

	_J.setZero();
	_JTJ.setZero();

	std::vector<std::array<Eigen::Matrix3d, 3>>vAllCam_dRdso3;
	vAllCam_dRdso3.resize(_iCameraSum);

	//_ResultCam[1].mR.at<double>(0, 0) = 0.993651;
	//_ResultCam[1].mR.at<double>(0, 1) = 0.00507611;
	//_ResultCam[1].mR.at<double>(0, 2) = -0.112391;
	//_ResultCam[1].mR.at<double>(1, 0) = -0.0079315;

	//_ResultCam[1].mR.at<double>(1, 1) = 0.999657;
	//_ResultCam[1].mR.at<double>(1, 2) = -0.0249733;
	//_ResultCam[1].mR.at<double>(2, 0) = 0.112226;
	//_ResultCam[1].mR.at<double>(2, 1) = 0.0257062;
	//_ResultCam[1].mR.at<double>(2, 2) = 0.99335;

	for (int i = 0; i < _iCameraSum; i++)
	{
		vAllCam_dRdso3[i] = dR_dso3(_ResultCam[i].mR);
	}
	///R关于so3的求导正确

	auto Homo_Eigen = [&](const cv::Mat Homo)
	{
		Eigen::Matrix3d H;
		H << Homo.at<double>(0, 0), Homo.at<double>(0, 1), Homo.at<double>(0, 2),
			Homo.at<double>(1, 0), Homo.at<double>(1, 1), Homo.at<double>(1, 2),
			Homo.at<double>(2, 0), Homo.at<double>(2, 1), Homo.at<double>(2, 2);
		return H;
	};

	for (int i = 0; i < _vMatchEdges.size(); i++)
	{
		const ConnectEdge_T& edge_pair = _vMatchEdges[i];
		int J_row = _vEdgesPts[i] * 2;						/*!< 当前边对应雅可比矩阵j,在整体J中的起始行 */

		///当前边端点对应的相机，在在整体J的列方向起始位置
		int iFrom = _vIndexMap_Org2Add[edge_pair.iFrom];
		int iTo = _vIndexMap_Org2Add[edge_pair.iTo];

		_ResultCam[iFrom].mK.at<double>(0, 2) = 0;
		_ResultCam[iFrom].mK.at<double>(1, 2) = 0;
		_ResultCam[iTo].mK.at<double>(0, 2) = 0;
		_ResultCam[iTo].mK.at<double>(1, 2) = 0;

		const auto& Cam_From = _ResultCam[iFrom];
		const auto& Cam_To = _ResultCam[iTo];
		//cout << "Cam_From.mK: " << Cam_From.mK << endl;

		const auto From_K_Mat = Cam_From.mK;
		const auto From_R_Mat = Cam_From.mR;
		const auto To_Kinv_Mat = Cam_To.mK.inv();
		const auto To_Rinv_Mat = Cam_To.mR.t();

		const auto From_K = Homo_Eigen(From_K_Mat);
		const auto From_R = Homo_Eigen(From_R_Mat);
		const auto To_Kinv = Homo_Eigen(To_Kinv_Mat);
		const auto To_Rinv = Homo_Eigen(To_Rinv_Mat);

		const auto H_FromTo = (From_K * From_R) * (To_Rinv * To_Kinv);

		const auto& dFrom_Rdso3 = vAllCam_dRdso3[iFrom];

		///dTo_Rdso3 = dFrom_Rdso3的转置
		auto dTo_Rdso3 = vAllCam_dRdso3[iTo];
		for (auto& m : dTo_Rdso3)
		{
			m = m.transpose().eval();
			//cout << m << endl;
			//m.transposeInPlace();
		}
		iFrom = iFrom * NR_PARAM_PER_CMAERA;
		iTo = iTo * NR_PARAM_PER_CMAERA;
		///遍历边的端点对应的匹配点
		for (const auto& p : edge_pair.EdgeMatchInfo.vMatchInlierPts)
		{
			cv::Point2f tempPt = p.second;
			//tempPt.x = 52.9667;
			//tempPt.y = -85.4211;
			//Eigen::Vector3d to{ p.second.x, p.second.y ,1 };
			Eigen::Vector3d to{ tempPt.x, tempPt.y ,1 };
			Eigen::Vector3d	to_H = H_FromTo * to;

			double dz_sqr_inv = 1.0 / pow(to_H.z(), 2.0);
			double dz_inv = 1.0 / to_H.z();

			Eigen::Vector3d drdp_;
#define drdv(xx) (drdp_ =xx,Eigen::Vector2d{-drdp_.x()*dz_inv+drdp_.z()*to_H.x()*dz_sqr_inv,-drdp_.y()*dz_inv+drdp_.z()*to_H.y()*dz_sqr_inv})

			array<Eigen::Vector2d, NR_PARAM_PER_CMAERA>dfrom_dcam, dto_dcam;

			///求解dr/dfrom
			///R*R_t*K_inv*to
			Eigen::Vector3d r_rinv_kinv_u = From_R * To_Rinv * To_Kinv * to;

			///dr/df
			dfrom_dcam[0] = drdv(DK_DFOCAL() * r_rinv_kinv_u);
			///dr/dcx
			dfrom_dcam[1] = drdv(DK_DCX() * r_rinv_kinv_u);
			///dr/dy
			dfrom_dcam[2] = drdv(DK_DCY() * r_rinv_kinv_u);

			///R_t* K_inv* to
			Eigen::Vector3d rinv_kinv_u = To_Rinv * To_Kinv * to;

			/// K*(dR/dso3(i))*R_t*K^inv*to: dr/dso3(0)| dr/dso3(1)| dr/dso3(2)
			dfrom_dcam[3] = drdv(From_K * dFrom_Rdso3[0] * rinv_kinv_u);
			dfrom_dcam[4] = drdv(From_K * dFrom_Rdso3[1] * rinv_kinv_u);
			dfrom_dcam[5] = drdv(From_K * dFrom_Rdso3[2] * rinv_kinv_u);

			///求解dr/dto
			///K*R*R^t*K^inv
			Eigen::Matrix3d k_r_rinv_kinv = From_K * From_R * To_Rinv * To_Kinv;
			///-K^inv*to
			Eigen::Vector3d _kinv_to = -To_Kinv * to;

			///dr/df  dr/dcx  dr/dcy
			dto_dcam[0] = drdv((k_r_rinv_kinv * DK_DFOCAL() * _kinv_to));
			dto_dcam[1] = drdv((k_r_rinv_kinv * DK_DCX() * _kinv_to));
			dto_dcam[2] = drdv((k_r_rinv_kinv * DK_DCY() * _kinv_to));

			Eigen::Matrix3d k_r = From_K * From_R;		/*!< K*R */

			Eigen::Vector3d kinv_to = To_Kinv * to;		/*!< K^inv*to */

			/// K*R*(dR/dso3(i))_T*K^inv*to:  dr/dso3(0)| dr/dso3(1)| dr/dso3(2)
			dto_dcam[3] = drdv(k_r * dTo_Rdso3[0] * kinv_to);
			dto_dcam[4] = drdv(k_r * dTo_Rdso3[1] * kinv_to);
			dto_dcam[5] = drdv(k_r * dTo_Rdso3[2] * kinv_to);
#undef drdv

			///填充当前边在J的部分;
			for (int k = 0; k < NR_PARAM_PER_CMAERA; k++)
			{
				_J(J_row, iFrom + k) = dfrom_dcam[k].x();
				_J(J_row, iTo + k) = dto_dcam[k].x();
				_J(J_row + 1, iFrom + k) = dfrom_dcam[k].y();
				_J(J_row + 1, iTo + k) = dto_dcam[k].y();
			}
			//cout << "_J: " << _J << endl;

			///填充当前边在JTJ中的部分
			///填充JTJ中 dfrom_dcam*dto_dcam的部分
			for (int m = 0; m < NR_PARAM_PER_CMAERA; m++)
			{
				int ir = iFrom + m;
				for (int n = 0; n < NR_PARAM_PER_CMAERA; n++)
				{
					int ic = iTo + n;
					auto val = dfrom_dcam[m].dot(dto_dcam[n]);
					_JTJ(ir, ic) += val;
					_JTJ(ic, ir) += val;
				}
			}
			///填充JTJ中 dfrom_dcam*dfrom_dcam | dto_dcam*dto_dcam的部分
			for (int m = 0; m < NR_PARAM_PER_CMAERA; m++)
			{
				for (int n = m; n < NR_PARAM_PER_CMAERA; n++)
				{
					///填充JTJ中 dfrom_dcam*dfrom_dcam
					int ir = iFrom + m;
					int ic = iFrom + n;
					auto val = dfrom_dcam[m].dot(dfrom_dcam[n]);
					_JTJ(ir, ic) += val;
					if (ir != ic)
					{
						_JTJ(ic, ir) += val;
					}

					///填充JTJ中 dfrom_dcam* dfrom_dcam
					ir = iTo + m;
					ic = iTo + n;
					val = dto_dcam[m].dot(dto_dcam[n]);
					_JTJ(ir, ic) += val;
					if (ir != ic)
					{
						_JTJ(ic, ir) += val;
					}
				}
			}
			J_row += 2;
		}
	}
}

///计算误差
void IncrementalBundleAdjuster::calcError(const Eigen::VectorXd& CamPara, Eigen::VectorXd& error)
{
	auto Proj_K = [&](const int& iID)
	{
		float fFocal, fCx, fCy;
		fFocal = CamPara[NR_PARAM_PER_CMAERA * iID + 0];
		fCx = CamPara[NR_PARAM_PER_CMAERA * iID + 1];
		fCy = CamPara[NR_PARAM_PER_CMAERA * iID + 2];

		cv::Mat K = OpenStitch::Camera::SoGetK(fFocal, fCx, fCy);
		return K;
	};

	auto Proj_R = [&](const int& iID)
	{
		cv::Mat Camso3 = cv::Mat::zeros(3, 1, CV_64F);
		Camso3.at<double>(0, 0) = CamPara[NR_PARAM_PER_CMAERA * iID + 3];
		Camso3.at<double>(1, 0) = CamPara[NR_PARAM_PER_CMAERA * iID + 4];
		Camso3.at<double>(2, 0) = CamPara[NR_PARAM_PER_CMAERA * iID + 5];
		cv::Mat R;
		OpenStitch::Camera::Soso32R(Camso3, R);
		return R;
	};

	auto pt_proj = [&](const double* ptrH, const cv::Point2f& pt)
	{
		double dX, dY, dZ;
		dX = ptrH[0] * pt.x + ptrH[1] * pt.y + ptrH[2] * 1;
		dY = ptrH[3] * pt.x + ptrH[4] * pt.y + ptrH[5] * 1;
		dZ = ptrH[6] * pt.x + ptrH[7] * pt.y + ptrH[8] * 1;
		double dZ_inv = 1 / dZ;
		return cv::Point2f(dX * dZ_inv, dY * dZ_inv);
	};
	int iIdx = 0;
	for (const auto& i : _vMatchEdges)
	{
		int iFrom = _vIndexMap_Org2Add[i.iFrom];
		int iTo = _vIndexMap_Org2Add[i.iTo];
		cv::Mat From_R, From_K, To_R, To_K;
		From_R = Proj_R(iFrom);
		From_K = Proj_K(iFrom);
		To_R = Proj_R(iTo);
		To_K = Proj_K(iTo);

		cv::Mat HomoFromTo = From_K * From_R * To_R.t() * To_K.inv();
		double ptrH[9] = { 0.0 };
		ptrH[0] = HomoFromTo.ptr<double>(0)[0];
		ptrH[1] = HomoFromTo.ptr<double>(0)[1];
		ptrH[2] = HomoFromTo.ptr<double>(0)[2];

		ptrH[3] = HomoFromTo.ptr<double>(1)[0];
		ptrH[4] = HomoFromTo.ptr<double>(1)[1];
		ptrH[5] = HomoFromTo.ptr<double>(1)[2];

		ptrH[6] = HomoFromTo.ptr<double>(2)[0];
		ptrH[7] = HomoFromTo.ptr<double>(2)[1];
		ptrH[8] = HomoFromTo.ptr<double>(2)[2];

		for (const auto& j : i.EdgeMatchInfo.vMatchInlierPts)
		{
			cv::Point2f to_pt, from_pt;
			from_pt = j.first;
			to_pt = j.second;
			cv::Point2f from_estimate = pt_proj(ptrH, to_pt);
			//_Error[iIdx] = from_pt.x - from_estimate.x;
			//_Error[iIdx + 1] = from_pt.y - from_estimate.y;

			error[iIdx] = from_pt.x - from_estimate.x;
			error[iIdx + 1] = from_pt.y - from_estimate.y;

			iIdx += 2;
		}
	}
	//cout << "_Error: " << _Error.maxCoeff() << endl;
}

void IncrementalBundleAdjuster::CalU(const Eigen::MatrixXd& JTJ)
{
	Eigen::VectorXd diagonal_vect = JTJ.diagonal();		/*!< 取对角线元素 */
	//cout << _J.transpose() * _J << endl;
	//cout << "_dU: " << diagonal_vect << endl;
	_dU = LM_PARA_TAU * diagonal_vect.maxCoeff();
}

double IncrementalBundleAdjuster::CalRho(const Eigen::VectorXd& CamNew)
{
	Eigen::VectorXd ErrorNew = Eigen::VectorXd::Zero(NR_TERM_PER_MATCH * _iTotalPtsSum);
	calcError(CamNew, ErrorNew);

	/// ||e||^2 - ||x - f(p_new)||^2
	double dNumerator = _Error.squaredNorm() - ErrorNew.squaredNorm();
	//cout << "_Error.squaredNorm(): " << _Error.squaredNorm() << endl;
	//cout << "ErrorNew.squaredNorm(): " << ErrorNew.squaredNorm() << endl;
	//double dNumerator = pow(_Error.squaredNorm(), 2) - pow(ErrorNew.squaredNorm(), 2);
	double dDenom = _CamIncrement.transpose() * (_dU * _CamIncrement + _G);
	double dRho = dNumerator / dDenom;
	return dRho;
}

void IncrementalBundleAdjuster::UpdateIndexMap()
{
	int iCnt = 0;
	for (auto i : _vOrgImgsID)
	{
		_vIndexMap_Org2Add[i] = iCnt++;
	}
}

void IncrementalBundleAdjuster::UpdateLMMat()
{
	calcJacobian(_J, _JTJ);
	calcError(_CamParaIteration, _Error);
	_G = -_J.transpose() * _Error;
	//_G = _J.transpose() * _Error;
}

///初始化相机参数
void IncrementalBundleAdjuster::InitCamaPara(const std::vector<OpenStitch::Camera>& vCamers)
{
	_CamParaIteration = Eigen::VectorXd{ NR_PARAM_PER_CMAERA * _iCameraSum };

	for (int i = 0; i < _iCameraSum; i++)
	{
		_CamParaIteration[NR_PARAM_PER_CMAERA * i + 0] = vCamers[i].mfFocal;
		_CamParaIteration[NR_PARAM_PER_CMAERA * i + 1] = vCamers[i].mfCx;
		_CamParaIteration[NR_PARAM_PER_CMAERA * i + 2] = vCamers[i].mfCy;

		cv::Mat Camso3 = cv::Mat::zeros(3, 1, CV_64F);
		OpenStitch::Camera::SoR2so3(vCamers[i].mR, Camso3);

		_CamParaIteration[NR_PARAM_PER_CMAERA * i + 3] = Camso3.at<double>(0, 0);
		_CamParaIteration[NR_PARAM_PER_CMAERA * i + 4] = Camso3.at<double>(1, 0);;
		_CamParaIteration[NR_PARAM_PER_CMAERA * i + 5] = Camso3.at<double>(2, 0);;
	}
}

void IncrementalBundleAdjuster::UpDateCamPara(std::vector<OpenStitch::Camera>& vCamers)
{
	for (int i = 0; i < _iCameraSum; i++)
	{
		float fFocal, fCx, fCy;
		fFocal = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 0];
		fCx = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 1];
		fCy = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 2];

		vCamers[i].mK = OpenStitch::Camera::SoGetK(fFocal, fCx, fCy);

		float so3_1, so3_2, so3_3;
		so3_1 = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 3];
		so3_2 = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 4];
		so3_3 = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 5];

		cv::Mat Camso3 = cv::Mat::zeros(3, 1, CV_64F);
		Camso3.at<double>(0, 0) = so3_1;
		Camso3.at<double>(1, 0) = so3_2;
		Camso3.at<double>(2, 0) = so3_3;

		OpenStitch::Camera::Soso32R(Camso3, vCamers[i].mR);
	}
}

///更新相机参数
void IncrementalBundleAdjuster::ObtainRefinedCameraParams(std::vector<OpenStitch::Camera>& vCamers)
{
	for (int i = 0; i < _iCameraSum; i++)
	{
		float fFocal, fCx, fCy;
		fFocal = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 0];
		fCx = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 1];
		fCy = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 2];
		vCamers[i].mfFocal = fFocal;
		vCamers[i].mfCx = fCx;
		vCamers[i].mfCy = fCy;

		vCamers[i].mK = OpenStitch::Camera::SoGetK(fFocal, fCx, fCy);

		cv::Mat Camso3 = cv::Mat::zeros(3, 1, CV_64F);
		Camso3.at<double>(0, 0) = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 3];
		Camso3.at<double>(1, 0) = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 4];
		Camso3.at<double>(2, 0) = _CamParaIteration[NR_PARAM_PER_CMAERA * i + 5];
		OpenStitch::Camera::Soso32R(Camso3, vCamers[i].mR);
	}
}

std::array<Eigen::Matrix3d, 3> IncrementalBundleAdjuster::dR_dso3(const cv::Mat& R)
{
	cv::Mat tempso3;
	OpenStitch::Camera::SoR2so3(R, tempso3);
	Eigen::Vector3d so3;
	so3[0] = tempso3.at<double>(0, 0);
	so3[1] = tempso3.at<double>(1, 0);
	so3[2] = tempso3.at<double>(2, 0);
	//cout << "R: " << R << endl;
	//cout << "so3: " << so3 << endl;

	double dso3_sqr = so3.squaredNorm();			/*!< 二范数 */
	if (dso3_sqr < 1e-15)
	{
		Eigen::Vector3d v1, v2, v3;
		v1 << 1, 0, 0;
		v2 << 0, 1, 0;
		v3 << 0, 0, 1;
		return std::array<Eigen::Matrix3d, 3>{
			SkewSymMetric(v1),
				SkewSymMetric(v2),
				SkewSymMetric(v3)};
	}
	Eigen::Matrix3d so3_Mat = SkewSymMetric(so3);			/*!< 生成反对称矩阵so3^ */
	std::array<Eigen::Matrix3d, 3>Theat_so3_Mat{ so3_Mat ,so3_Mat ,so3_Mat };				/*!< Theat(i)*so3_Mat */
	for (int i = 0; i < 3; i++)
	{
		Theat_so3_Mat[i] *= so3[i];
	}
	Eigen::Vector3d I_R_e;						/*!< (I-R)e(i) */
	I_R_e << 1 - R.at<double>(0, 0), -R.at<double>(1, 0), -R.at<double>(2, 0);
	I_R_e = so3.cross(I_R_e);					/*!< so3 叉乘 (I-R)*e(0) */
	Theat_so3_Mat[0] += SkewSymMetric(I_R_e);	/*!< Theat(0)*so3_Mat + (I_R_e(0))^ */

	I_R_e << -R.at<double>(0, 1), 1 - R.at<double>(1, 1), -R.at<double>(2, 1);
	I_R_e = so3.cross(I_R_e);					/*!< so3 叉乘 (I-R)*e(1) */
	Theat_so3_Mat[1] += SkewSymMetric(I_R_e);	/*!< Theat(1)* so3_Mat + (I_R_e(1))^ */

	I_R_e << -R.at<double>(0, 2), -R.at<double>(1, 2), 1 - R.at<double>(2, 2);
	I_R_e = so3.cross(I_R_e);					/*!< so3 叉乘 (I-R)*e(2) */
	Theat_so3_Mat[2] += SkewSymMetric(I_R_e);	/*!< Theat(2)* so3_Mat + (I_R_e(2))^ */

	Eigen::Matrix3d TempR;
	TempR << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
	int iIndex = 0;
	for (auto& i : Theat_so3_Mat)
	{
		i = i * (1 / dso3_sqr);
		i = i * TempR;
		//cout << iIndex << ": " << i << endl;
		iIndex++;
	}
	return Theat_so3_Mat;
}