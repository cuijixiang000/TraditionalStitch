#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/util.hpp>

#include <iostream>

#include "TransStitcher.h"
#include "CylinderStitcher.h"
#include "EstimateStitcher.h"

#include "CylinderWarp.h"

#include "StitchBase.h"
#include "common.h"

using namespace cv;

bool RealAllFolder(const string& strInputPath, vector<string>& _vFilePath);

string strImgName(const string& strVideoFile);

void findMaxSpanningTree111(int num_images, const vector<cv::detail::MatchesInfo>& pairwise_matches, cv::detail::Graph& span_tree, vector<int>& centers);

void FindTree();

int main(int argc, char** argv)
{
	//FindTree();

	//std::set<int>idx_added;
	//int ptra[3] = { 10,10,10 };
	//int ptrb[3] = { 11,12,13 };
	//for (int i = 0; i < 3; i++)
	//{
	//	idx_added.insert(ptra[i]);
	//	idx_added.insert(ptrb[i]);
	//}

	string strInputPath = "../data1";

	string OutPutFolder = "../TestResultCy/BA/";

	vector<string> _vFolderPath;
	RealAllFolder(strInputPath, _vFolderPath);

	StitchConfig Config;
	Config.iPanoModel = PanoModel_M::PANO_CYLINDRICAL;
	Config.iFeatureMethod = FeatureMethod_M::FEATURE_SURF;
	Config.iMatchMethod = MatchMethod_M::MATCH_FLANN;
	Config.iBlenderMethod = 0;
	Config.iOrder = 0;
	Config.fCamFocal = 30;  /*!< 焦距比较重要mm */
	Config.TransformType = TransModel_M::HOMO;

	SoInitLogFile("PanoLogs");

	int iIndex = 0;
	for (auto i : _vFolderPath)
	{
		string strFolderName = strImgName(i);
		std::vector<string>vImgsPath;
		string strImgsPath = strInputPath + "/SPHP-park/";
		cv::glob(strImgsPath, vImgsPath);

		//cv::glob(i, vImgsPath);
		std::vector<cv::Mat>vImgs;
		for (auto j : vImgsPath)
		{
			cv::Mat Img = cv::imread(j);
			vImgs.emplace_back(Img);
		}

		std::shared_ptr<StitchBase>classTransStitcher;

		///平移模式
		//classTransStitcher = std::make_shared<TransStitcher>(vImgs, Config);

		///柱面模式
		//classTransStitcher = std::make_shared<CylinderStitcher>(vImgs, Config);

		///光束法平差模式
		classTransStitcher = std::make_shared<EstimateStitcher>(vImgs, Config);

		cv::Mat PanoImg = classTransStitcher->SoBuild();

		//cv::Mat PanoImg;
		if (!PanoImg.data)
		{
			cout << i << " is error!!!" << endl;
			continue;
		}
		string strSavePath = OutPutFolder + strFolderName + "-BA.png";
		cv::imwrite(strSavePath, PanoImg);
		cout << i << " is OK!!!" << endl;
		vImgs.clear();
		return 0;
	}

	//Ptr<Stitcher> stitcher = Stitcher::create();
	//Mat pano;
	//Stitcher::Status status = stitcher->stitch(vImgs, pano);
	//if (status != Stitcher::OK)
	//{
	//	cout << "Can't stitch images, error code = " << int(status) << endl;
	//	return -1;
	//}
	//cv::imwrite("../TestResult/PanoImg.png", pano);
	//return 0;

	return 0;
}

bool RealAllFolder(const string& strInputPath, vector<string>& _vFilePath)
{
	///读取文件夹内所有文件
	long long  hFile = 0;
	//文件信息
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(strInputPath).append("/*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之
			//如果不是,加入列表
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					string ImgName = p.assign(strInputPath).append("/").append(fileinfo.name);
					_vFilePath.push_back(ImgName);
					//	RealAllFolder(ImgName,/* _strImgName,*/ _vFilePath);
				}
				//	RealAllFolder(p.assign(strInputPath).append("/").append(fileinfo.name),/* _strImgName,*/ _vFilePath);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	else
	{
		return false;
	}
	return true;
}

string strImgName(const string& strVideoFile)
{
	int iStrat, iEnd;
	iEnd = strVideoFile.find_last_of("/");
	int iLength = strVideoFile.length();
	//iStrat = strVideoFile.find_last_of("\\");
	string strName;
	strName = strVideoFile.substr(iEnd + 1, iLength - iEnd);
	return strName;
}

void findMaxSpanningTree111(int num_images, const vector<cv::detail::MatchesInfo>& pairwise_matches, cv::detail::Graph& span_tree, vector<int>& centers)
//num_images表示待拼接图像的数量，也是最大生成树的节点数
//pairwise_matches表示图像间的拼接信息
//span_tree表示最大生成树
//centers表示最大生成树的中心节点
{
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
			if (pairwise_matches[i * num_images + j].H.empty())
				continue;
			//得到图像i和j的内点数
			float conf = static_cast<float>(pairwise_matches[i * num_images + j].num_inliers);
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

void FindTree()
{
	int num_images;

	cv::detail::Graph span_tree;
	vector<int> centers;
	num_images = 5;
	vector<cv::detail::MatchesInfo> pairwise_matches(num_images * num_images);
	pairwise_matches[0 * num_images + 1].H = cv::Mat::eye(3, 3, CV_64F);
	pairwise_matches[0 * num_images + 1].num_inliers = 60;

	pairwise_matches[0 * num_images + 2].H = cv::Mat::eye(3, 3, CV_64F);
	pairwise_matches[0 * num_images + 2].num_inliers = 50;

	pairwise_matches[1 * num_images + 2].H = cv::Mat::eye(3, 3, CV_64F);
	pairwise_matches[1 * num_images + 2].num_inliers = 40;

	pairwise_matches[2 * num_images + 3].H = cv::Mat::eye(3, 3, CV_64F);
	pairwise_matches[2 * num_images + 3].num_inliers = 20;

	pairwise_matches[2 * num_images + 4].H = cv::Mat::eye(3, 3, CV_64F);
	pairwise_matches[2 * num_images + 4].num_inliers = 30;

	struct Edge {
		int v1, v2;
		float weight;
		Edge(int a, int b, float v) :v1(a), v2(b), weight(v) {}
		//Edge(float v) :weight(v) {}
		//Edge(int a, int b) :v1(a), v2(b) {}
		bool operator < (const Edge& r) const { return weight < r.weight; }   /*!< 升序排列吗？ */
	};
	priority_queue<Edge>vp;
	for (int i = 0; i < 1; i++)
	{
		float aa = float(i);
		for (int j = i; j < 5; j++)
		{
			Edge tempdata(0, j, j);
			//tempdata.v1 = i, tempdata.v2 = j, tempdata.weight = j;
			vp.emplace(tempdata);
		}
	}
	vp.emplace(0, 3, 4.0);
	findMaxSpanningTree111(num_images, pairwise_matches, span_tree, centers);
}