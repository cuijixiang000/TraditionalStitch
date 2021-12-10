# 欢迎使用`OpenStitch`全景拼接库! :) ;)
>如果您发现任何问题,欢迎至邮件给我们提出您宝贵的意见,我们会尽快修复!

## 简介	
OpenStitch是一个全景拼接算法库,主要分为两个主要方向:
* **基于深度学习的图像拼接**
	* 基于单应性估计的方法:[Content-Aware Unsupervised Deep Homography Estimation](https://github.com/JirongZhang/DeepHomography)
	* 基于多格网估计的方法:[DeepMeshFlow: Content Adaptive Mesh Deformation for Robust Image
Registration](https://deepai.org/publication/deepmeshflow-content-adaptive-mesh-deformation-for-robust-image-registration)
* **基于传统发给发的图像拼接**
	* 基于单应性矩阵的全景拼接:[Automatic Panoramic Image Stitching using Invariant Features](http://matthewalunbrown.com/papers/ijcv2007.pdf)
	其中基于单应行矩阵的全景拼接又分为三个模式:***平面模型***、***柱面模型***、***相机估计模型(BA+球面投影)***.主要参考[OpenPanp](http://https://github.com/ppwwyyxx/OpenPano)进行修改与扩充.
	* 基于多个网优化的全景拼接:[Natural Image Stitching with the Global Similarity Prior](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_12P)
	主要参考[NISwGSP](https://github.com/nothinglo/NISwGSP)进行修改与扩充.
    
## 编译依赖:
 
* gcc >= 4.7 (Or  >= VS2017)
```
sudo apt-get install g++
```
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)
```
sudo apt-get install libeigen3-dev

```
* [Opencv](http://https://github.com/opencv/opencv)	 >=4.0
```
sudo apt-get install libopencv-dev   #默认安装Ubuntu对应版本的OpenCV,Ubuntu20默认4.0以上版本
```
* [Cmake](https://cmake.org/download/) >=3.1.0
```
sudo apt-get install cmake     #安装Ubuntu自带的Cmake
```

## TraditionalStitch编译

#### Linux /WSL
```
$ mkdir build && cd build && cmake .. && make
```
#### Windows
* 安装 *Cmake*、*VS2017*
* 设置环境变量`Eigen3_DIR`、`OpenCV_DIR`
* Cmake->Configure->Generate->Open Project
