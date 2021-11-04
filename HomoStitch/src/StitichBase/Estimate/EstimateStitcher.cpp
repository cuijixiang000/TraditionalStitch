#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/util.hpp>

#include <queue>
#include <iostream>

#include "Camera.h"
#include "IncrementalBundleAdjuster.h"

#include "EstimateStitcher.h"

#ifndef CALC_ROTATION_T
#define	CALC_ROTATION_T
struct CalcRotation			/*!< 根据opencv源码改写 */
{
	CalcRotation(int _num_images, const vector<vector<MatchInfo_T>>& _pairwise_matches, vector<OpenStitch::Camera>& _cameras)
		: num_images(_num_images), pairwise_matches(_pairwise_matches), cameras(&_cameras[0]) {}

	void operator ()(const cv::detail::GraphEdge& edge)
	{
		int pair_idx = edge.from * num_images + edge.to;    //表示匹配点对的索引
		//构造式51中的参数K0
		cv::Mat_<double> K_from = cv::Mat::eye(3, 3, CV_64F);    //初始化
		K_from(0, 0) = cameras[edge.from].mfFocal;    //表示式33的fu
		//表示式33的fv
		K_from(1, 1) = cameras[edge.from].mfFocal;
		K_from(0, 2) = cameras[edge.from].mfCx;    //表示式33的cx
		K_from(1, 2) = cameras[edge.from].mfCy;    //表示式33的cy
		//构造式51中的参数K1
		cv::Mat_<double> K_to = cv::Mat::eye(3, 3, CV_64F);    //初始化
		K_to(0, 0) = cameras[edge.to].mfFocal;    //表示式33的fu
		K_to(1, 1) = cameras[edge.to].mfFocal;
		K_to(0, 2) = cameras[edge.to].mfCx;    //表示式33的cx
		K_to(1, 2) = cameras[edge.to].mfCy;    //表示式33的cy

		//cv::Mat R = K_from.inv() * pairwise_matches[edge.from][edge.to].Homo.inv() * K_to;    //式51

		cv::Mat R = K_from.inv() * pairwise_matches[edge.from][edge.to].Homo * K_to;
		//式52，可见CameraParams变量中R实际存储的是相机旋转矩阵变量的逆
		cameras[edge.to].mR = (cameras[edge.from].mR.t() * R).t();
		//cameras[edge.to].mR = (cameras[edge.from].mR.t() * R);
		cameras[edge.to].mK = K_to.clone();

		cameras[edge.from].mK = K_from.clone();
	}

	int num_images;    //表示待拼接图像的数量
	const vector < vector<MatchInfo_T>>& pairwise_matches;    //表示匹配图像的信息
	OpenStitch::Camera* cameras;    //表示相机参数
};
#endif		/*!< CALC_ROTATION_T */

cv::Mat EstimateStitcher::SoBuild()
{
	LOG("********Start EstimateStitcher...********");

	CalFeatures();
	LOG("********@CalFeatures@ is OK********");

	if (!EstimateFocal())
	{
		ERROR("EstimateImgsMatch ERROR!!!");
		return cv::Mat();
	}
	LOG("********@EstimateFocal@ is OK********");

	SetInitCameraParams();
	LOG("********@SetInitCameraParams@ is OK********");

	Optimize();
	LOG("********@Optimize@ is OK********");

	UpDateRange();
	LOG("********@UpDateRange@ is OK********");

	LOG("********EstimateStitcher is OK********");
	return  Blender();
}

bool EstimateStitcher::EstimateFocal()
{
	bool bSuss = false;
	int iImgsNum = m_vImgs.size();

	vector<double> all_focals;

	_vvMatchsInfo.resize(iImgsNum);
	for (auto& i : _vvMatchsInfo)
	{
		i.resize(iImgsNum);
	}
	vector<pair<int, int>>vtasks;
	for (int i = 0; i < iImgsNum; i++)
	{
		//int iNext = (i + 1) % iImgsNum;

		//if (i == iImgsNum - 1)
		//{
		//	continue;
		//}
		//vtasks.emplace_back(i, iNext);
		for (int j = i + 1; j < iImgsNum; j++)
		{
			vtasks.emplace_back(i, j);
		}
	}
#pragma omp parallel for schedule(dynamic)
	for (int k = 0; k < vtasks.size(); k++)
	{
		int i = vtasks[k].first;
		int j = vtasks[k].second;

		MatchInfo_T ImgMathInfo;
		ImgMathInfo.MatchShape.first = m_vImgs[i].size();
		ImgMathInfo.MatchShape.second = m_vImgs[j].size();

		bSuss = PairWiseMatch(m_vvKeyPoints[i], m_vDstps[i], m_vvKeyPoints[j], m_vDstps[j], ImgMathInfo);
		////continue;

		//DEBUG("Image {} and {} has {} matched points", i, j, ImgMathInfo.iInlierSize);
		if (!bSuss)	continue;

		bSuss = HFilter(ImgMathInfo);
		//continue;

		if (!bSuss)	continue;
		DEBUG("Image {} and {} has {} matched points", i, j, ImgMathInfo.iInlierSize);
#pragma omp critical
		{
			_vvMatchsInfo[i][j] = ImgMathInfo;
		}
		//_vvMatchsInfo[i][j] = ImgMathInfo;
		//ImgMathInfo.vMatchPair.clear();

		///根据H估算焦距
		float f0, f1;
		bool bf0, bf1;
		FindfocalsFromHomography(ImgMathInfo.Homo, f0, f1, bf0, bf1);
		if (bf0 && bf1)
		{
			all_focals.emplace_back(sqrt(f0 * f1));
		}

		///判断是否可逆
		cv::Mat Inv = ImgMathInfo.Homo.inv();
		Inv *= (1.0 / Inv.at<double>(2, 2));

		ImgMathInfo.Homo = Inv;
		ImgMathInfo.reverse();

		///加线程锁？
#pragma omp critical
		{
			_vvMatchsInfo[j][i] = move(ImgMathInfo);
		}
		//_vvMatchsInfo[j][i] = move(ImgMathInfo);

		//ImgMathInfo.vKptsI.clear();
		//ImgMathInfo.vKptsJ.clear();
		//ImgMathInfo.viMatchIndex.clear();
	}

	_vCameras.resize(iImgsNum);
	for (auto& i : _vCameras)
	{
		i.mR = cv::Mat::eye(3, 3, CV_64F);
		i.mK = cv::Mat::eye(3, 3, CV_64F);
	}

	//	cout << "**********估算相机焦距**********" << endl;
		//return false;
		///取中值
	if (static_cast<int>(all_focals.size()) >= iImgsNum - 1)
	{
		double median;
		std::sort(all_focals.begin(), all_focals.end());
		if (all_focals.size() % 2 == 1)
			median = all_focals[all_focals.size() / 2];
		else
			median = (all_focals[all_focals.size() / 2 - 1] + all_focals[all_focals.size() / 2]) * 0.5;
		for (int i = 0; i < iImgsNum; ++i)
			_vCameras[i].mfFocal = median;
		//_vCameras[i].mfFocal = 1033.795396;
		return true;
	}
	else
	{
		LOG("Can't estimate focal length, will use naive approach");
		double focals_sum = 0;
		for (int i = 0; i < iImgsNum; ++i)    //所有图像的长宽之和
			focals_sum += m_vImgs[i].cols + m_vImgs[i].rows;
		for (int i = 0; i < iImgsNum; ++i)
			_vCameras[i].mfFocal = focals_sum / iImgsNum;    //平均值
			//_vCameras[i].mfFocal = 1033.795396;
		return true;
	}
}
void EstimateStitcher::UpDateRange()
{
	///计算出来的是弧度角
	m_proj_method = ProjMed::spherical;
	auto homo2proj = get_homo2proj();
	auto pt_proj = [&](const double* ptrH, const cv::Point2f& pt)
	{
		double dX, dY, dZ;
		dX = ptrH[0] * pt.x + ptrH[1] * pt.y + ptrH[2] * 1;
		dY = ptrH[3] * pt.x + ptrH[4] * pt.y + ptrH[5] * 1;
		dZ = ptrH[6] * pt.x + ptrH[7] * pt.y + ptrH[8] * 1;
		cv::Point3f p3d(dX, dY, dZ);
		cv::Point2f tempPt;
		tempPt = homo2proj(p3d);
		return tempPt;
	};

	///图像边缘间隔采样
	std::vector<cv::Point2f>vEdgeCorners;
	EdgeSampling(vEdgeCorners);
	auto cal_range = [&](const cv::Mat& Homo, const cv::Size& ImgSize)
	{
		ImgRage_T img_range;
		double ptrH[9] = { 0.0 };
		ptrH[0] = Homo.ptr<double>(0)[0];
		ptrH[1] = Homo.ptr<double>(0)[1];
		ptrH[2] = Homo.ptr<double>(0)[2];

		ptrH[3] = Homo.ptr<double>(1)[0];
		ptrH[4] = Homo.ptr<double>(1)[1];
		ptrH[5] = Homo.ptr<double>(1)[2];

		ptrH[6] = Homo.ptr<double>(2)[0];
		ptrH[7] = Homo.ptr<double>(2)[1];
		ptrH[8] = Homo.ptr<double>(2)[2];

		for (const auto& v : vEdgeCorners)
		{
			cv::Point2f tempdata;
			tempdata.x = v.x * ImgSize.width;
			tempdata.y = v.y * ImgSize.height;
			cv::Point2f t_corner;

			t_corner = pt_proj(ptrH, tempdata);

			img_range.MinXY.x = MIN(img_range.MinXY.x, t_corner.x);
			img_range.MinXY.y = MIN(img_range.MinXY.y, t_corner.y);

			img_range.MaxXY.x = MAX(img_range.MaxXY.x, t_corner.x);
			img_range.MaxXY.y = MAX(img_range.MaxXY.y, t_corner.y);
		}
		return img_range;
	};

	int iIndex = 0;

	for (auto& i : m_GroupImgs.vImagesRane)
	{
		//std::cout << "Homo: " << i.HPanoSrc << std::endl;
		i.SingleImgRange = cal_range(i.HPanoSrc, m_vImgs[iIndex].size());

		DEBUG("projection range:({},{})~({},{})", i.SingleImgRange.MinXY.x, i.SingleImgRange.MinXY.y, i.SingleImgRange.MaxXY.x, i.SingleImgRange.MaxXY.y);

		//cout << "projection range: " << "(" << i.SingleImgRange.MinXY << ")~~~" << "(" << i.SingleImgRange.MaxXY << ")" << endl;

		iIndex++;
		m_GroupImgs.PanoRange.MaxXY.x = MAX(m_GroupImgs.PanoRange.MaxXY.x, i.SingleImgRange.MaxXY.x);
		m_GroupImgs.PanoRange.MaxXY.y = MAX(m_GroupImgs.PanoRange.MaxXY.y, i.SingleImgRange.MaxXY.y);

		m_GroupImgs.PanoRange.MinXY.x = MIN(m_GroupImgs.PanoRange.MinXY.x, i.SingleImgRange.MinXY.x);
		m_GroupImgs.PanoRange.MinXY.y = MIN(m_GroupImgs.PanoRange.MinXY.y, i.SingleImgRange.MinXY.y);
	}

	cv::Mat Homo_Center = m_GroupImgs.vImagesRane[_iCenterNode].HPanoSrc;
	vEdgeCorners.clear();

	vEdgeCorners.emplace_back(cv::Point2f(0, 0));
	vEdgeCorners.emplace_back(cv::Point2f(m_vImgs[_iCenterNode].size().width, m_vImgs[_iCenterNode].size().height));
	auto temprange = cal_range(Homo_Center, cv::Size(1, 1));
	_CenterImgSize.x = temprange.MaxXY.x - temprange.MinXY.x;
	_CenterImgSize.y = temprange.MaxXY.y - temprange.MinXY.y;
	//cout << "Identity projection range: " << "(" << temprange.MinXY << ")~~~" << "(" << temprange.MaxXY << ")" << endl;
	if (_CenterImgSize.x < 0)
	{
		_CenterImgSize.x = 2 * CV_PI + _CenterImgSize.x;
	}
	if (_CenterImgSize.y < 0)
	{
		_CenterImgSize.y = CV_PI + _CenterImgSize.y;
	}
}

void EstimateStitcher::Optimize()
{
	std::shared_ptr<IncrementalBundleAdjuster>ptrBA = std::make_shared<IncrementalBundleAdjuster>(_vCameras);
	ptrBA->SoSetCenterID(_iCenterNode);
	int iIndex = 0;
	//for (const auto i : _vCameras)
	//{
	//	cout << iIndex << ": K " << i.mK << endl;
	//	cout << iIndex << ": R " << i.mR << endl;
	//	iIndex++;
	//}

	int num_images = m_vImgs.size();

	int iEdges = 0;
	for (int i = 0; i < num_images; i++)
		//for (int i = 1; i < num_images; i++)
	{
		//for (int j = 0; j < i; j++)
		for (int j = i + 1; j < num_images; j++)
		{
			if (_vvMatchsInfo[i][j].fConfidence > 0)
			{
				ptrBA->SoAddEdge(i, j, _vvMatchsInfo[i][j]);
				iEdges++;
			}
		}
	}
	if (0 == iEdges)
	{
		ERROR("@SoAddEdge@ iEdges{}", iEdges);
		exit(1);
	}
	ptrBA->SoLMOptimize();
	//ptrBA->SoGNOptimize();

	///矩阵赋值
	m_GroupImgs.vImagesRane.clear();
	for (const auto& i : _vCameras)
	{
		cv::Mat Homo, Homo_Inv;
		Homo_Inv = i.mK * i.mR;
		Homo = i.mR.t() * i.mK.inv();
		MatchComponent_T tempdata;
		tempdata.HPanoSrc = Homo;
		tempdata.HSrcPano = Homo_Inv;
		//std::cout << "Homo: " << Homo << std::endl;
		//std::cout << "Homo_Inv: " << Homo_Inv << std::endl;
		m_GroupImgs.vImagesRane.emplace_back(tempdata);
		//std::cout << "Homo: " << tempdata.HPanoSrc << std::endl;
		//std::cout << "Homo_Inv: " << tempdata.HSrcPano << std::endl;
	}

	//for (auto i : m_GroupImgs.vImagesRane)
	//{
	//	std::cout << "Homo: " << i.HPanoSrc << std::endl;
	//}
	///添加边
//ptrBA->SoAddEdge();
//ptrBA->SoLMOptimize();			/*!< LM迭代 */
//ptrBA->SoGNOptimize();			/*!< 高斯牛顿迭代 */
}

void EstimateStitcher::SetInitCameraParams()
{
	int num_images = m_vImgs.size();
	_vCameras.resize(num_images);
	for (int i = 0; i < num_images; i++)
	{
		_vCameras[i].mfCx = 0.5 * m_vImgs[i].cols;
		_vCameras[i].mfCy = 0.5 * m_vImgs[i].rows;
	}

	///扣的opencv源码
	cv::detail::Graph span_tree;
	vector<int> centers;
	findMaxSpanningTree(num_images, _vvMatchsInfo, span_tree, centers);
	_iCenterNode = centers[0];
	span_tree.walkBreadthFirst(_iCenterNode, CalcRotation(num_images, _vvMatchsInfo, _vCameras));
	cout << "findMaxSpanningTree: " << _iCenterNode << endl;
	LOG("findMaxSpanningTree: {}", _iCenterNode);
}

void EstimateStitcher::findMaxSpanningTree(int num_images, const vector<vector<MatchInfo_T>>& pairwise_matches, cv::detail::Graph& span_tree, vector<int>& centers)
{
	//num_images表示待拼接图像的数量，也是最大生成树的节点数
//pairwise_matches表示图像间的拼接信息
//span_tree表示最大生成树
//centers表示最大生成树的中心节点

	struct IncDistance
	{
		IncDistance(std::vector<int>& vdists) : dists(&vdists[0]) {}
		void operator ()(const cv::detail::GraphEdge& edge) { dists[edge.to] = dists[edge.from] + 1; }
		int* dists;
	};

	cv::detail::Graph graph(num_images);    //定义无向图G
	vector<cv::detail::GraphEdge> edges;    //定义无向图G的边

	// Construct images graph and remember its edges
	//遍历待拼接图像，得到无向图G的边，即内点数
	for (int i = 0; i < num_images; ++i)
	{
		for (int j = 0; j < num_images; ++j)
		{
			//如果图像i和j没有单应矩阵，则说明这两幅图像不重叠，不能拼接
			if (pairwise_matches[i][j].Homo.empty())
				continue;
			//得到图像i和j的内点数
			//float conf = static_cast<float>(pairwise_matches[i][j].iInlierSize);
			float conf = static_cast<float>(pairwise_matches[i][j].fConfidence);
			graph.addEdge(i, j, conf);    //为G添加边
			edges.push_back(cv::detail::GraphEdge(i, j, conf));    //添加到边队列中
		}
	}

	cv::detail::DisjointSets comps(num_images);    //实例化DisjointSets类，表示定义一个并查集
	span_tree.create(num_images);    //创建生成树
	//表示生成树的幂，即节点间的连接数，例如某节点的幂为3，说明该节点与其他3个节点相连接
	vector<int> span_tree_powers(num_images, 0);

	// Find maximum spanning tree
	//按无向图G的边的大小（内点数）从小到大排序
	sort(edges.begin(), edges.end(), greater<cv::detail::GraphEdge>());
	for (size_t i = 0; i < edges.size(); ++i)    //从小到大遍历无向图G的边
	{
		int comp1 = comps.findSetByElem(edges[i].from);    //得到该边的起始节点的集合
		int comp2 = comps.findSetByElem(edges[i].to);    //得到该边的终止节点的集合
		//两种不相等，说明是一个新的边，需要通过并查集添加到生成树中
		if (comp1 != comp2)
		{
			comps.mergeSets(comp1, comp2);    //合并这两个节点
			//为生成树添加该边
			span_tree.addEdge(edges[i].from, edges[i].to, edges[i].weight);
			span_tree.addEdge(edges[i].to, edges[i].from, edges[i].weight);
			//节点幂的累加
			span_tree_powers[edges[i].from]++;
			span_tree_powers[edges[i].to]++;
		}
	}

	// Find spanning tree leafs
	vector<int> span_tree_leafs;    //表示生成树的叶节点
	//生成树的节点的幂为1，则为叶节点
	for (int i = 0; i < num_images; ++i)    //遍历图像
		if (span_tree_powers[i] == 1)    //表示该图像为叶节点
			span_tree_leafs.push_back(i);    //放入队列中

	// Find maximum distance from each spanning tree vertex
	vector<int> max_dists(num_images, 0);    //表示节点与叶节点的最大距离
	vector<int> cur_dists;    //表示节点与叶节点的当前距离
	for (size_t i = 0; i < span_tree_leafs.size(); ++i)    //遍历叶节点
	{
		cur_dists.assign(num_images, 0);    //初始化
		//得到该叶节点到其他节点的距离，IncDistance表示距离的累加，即节点的累计
		span_tree.walkBreadthFirst(span_tree_leafs[i], IncDistance(cur_dists));
		//遍历所有节点，更新节点到叶节点的最大距离
		for (int j = 0; j < num_images; ++j)
			max_dists[j] = max(max_dists[j], cur_dists[j]);
	}

	// Find min-max distance
	int min_max_dist = max_dists[0];    //表示所有最大距离中的最小值
	for (int i = 1; i < num_images; ++i)    //遍历所有节点
		if (min_max_dist > max_dists[i])
			min_max_dist = max_dists[i];    //得到最大距离中的最小值

	// Find spanning tree centers
	centers.clear();    //表示中心节点，清零
	for (int i = 0; i < num_images; ++i)    //遍历所有节点
		if (max_dists[i] == min_max_dist)
			centers.push_back(i);    //保存最大距离中的最小值所对应的节点
	//确保中心节点的数量必须大于0并小于3
	CV_Assert(centers.size() > 0 && centers.size() <= 2);
}

void EstimateStitcher::InitCenterNode(const int& iCenterNode)
{
}

void EstimateStitcher::InitNodePara(const int& iFrom, const int& iTo)
{
}

void EstimateStitcher::FindfocalsFromHomography(const cv::Mat& H, float& f0, float& f1, bool& bf0_ok, bool& bf1_ok)
{
	//确保H的数据类型和格式正确
	CV_Assert(H.type() == CV_64F && H.size() == cv::Size(3, 3));

	const double* h = reinterpret_cast<const double*>(H.data);    //赋值指针

	double d1, d2; // Denominators

	double v1, v2; // Focal squares value candidates

	bf1_ok = true;
	d1 = h[6] * h[7];
	d2 = (h[7] - h[6]) * (h[7] + h[6]);
	v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;
	v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;
	if (v1 < v2) std::swap(v1, v2);

	if (v1 > 0 && v2 > 0) f1 = sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f1 = sqrt(v1);
	else bf1_ok = false;

	bf0_ok = true;
	d1 = h[0] * h[3] + h[1] * h[4];
	d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
	v1 = -h[2] * h[5] / d1;
	v2 = (h[5] * h[5] - h[2] * h[2]) / d2;
	if (v1 < v2) std::swap(v1, v2);

	if (v1 > 0 && v2 > 0) f0 = sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0)
	{
		f0 = sqrt(v1);
	}
	else bf0_ok = false;
	if (std::isinf(f1) || std::isinf(f0))
	{
		bf1_ok = bf0_ok = false;
	}
}

cv::Point2f EstimateStitcher::GetFinalResolusion()
{
	cv::Point2f resolusion;
	if (ProjMed::flat == m_proj_method || ProjMed::cylindrical == m_proj_method)
	{
		resolusion.x = resolusion.y = 1;
	}
	else
	{
		///按照基准图像投影的弧度范围与图像真实尺寸，计算分辨率
		cv::Size BaseImgSize = m_vImgs[_iCenterNode].size();
		///计算基准图像对应的弧度范围
		resolusion.x = _CenterImgSize.x / BaseImgSize.width;
		resolusion.y = _CenterImgSize.y / BaseImgSize.height;
		///计算分辨率

		cv::Size PanoSize;
		PanoSize.width = m_GroupImgs.PanoRange.MaxXY.x - m_GroupImgs.PanoRange.MinXY.x + 1;
		PanoSize.height = m_GroupImgs.PanoRange.MaxXY.y = m_GroupImgs.PanoRange.MinXY.y + 1;

		PanoSize.width /= resolusion.x;
		PanoSize.height /= resolusion.y;

		double dMaxEdge = MAX(PanoSize.width, PanoSize.height);

		if (dMaxEdge > 80000 || PanoSize.width * PanoSize.height > 1e9)
		{
			///反馈可能拼接错误的消息
		}
		if (dMaxEdge > _iMaxPanoSize)
		{
			float fRatio = dMaxEdge / _iMaxPanoSize;
			resolusion *= fRatio;
		}
	}
	return resolusion;
}