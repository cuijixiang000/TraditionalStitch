#ifndef BLEND_BASE_H
#define BLEND_BASE_H

#include <opencv2/core.hpp>

#include <functional>

class BlenderBase
{
public:

	virtual void SoAddBlenderImgs(const cv::Point2f& upper_left, const cv::Point2f& bottom_right, \
		const cv::Mat& img, \
		std::function<cv::Point2f(cv::Point2i)>coor_func) = 0;

	virtual cv::Mat SoRunBlender(const int& iImgOrder) = 0;

protected:

	///双线性插值
	cv::Point3f interpolate(const cv::Mat& Img, const cv::Point2f& Pano2ImgUV);

	///图像裁切
public:

	struct Range {
		cv::Point2f min, max;	// min, max are both inclusive
		bool contain(int r, int c) const {
			return (r >= min.y && r <= max.y
				&& c >= min.x && c <= max.x);
		}
		int width() const { return max.x - min.x + 1; }
		int height() const { return max.y - min.y + 1; }

		friend std::ostream& operator << (std::ostream& os, const Range& s) {
			os << "min=" << s.min << ",max=" << s.max;
			return os;
		}
	};

	struct ImageToAdd {
		Range range;
		const cv::Mat imgref;
		std::function<cv::Point2f(cv::Point2i)> coor_func;

		cv::Point2f map_coor(int r, int c) const {
			auto ret = coor_func(cv::Point2i(c, r));
			if (ret.x < 0 || ret.x >= imgref.cols || ret.y < 0 || ret.y >= imgref.rows)
				ret = cv::Point2f(-999, -999);
			return ret;
		}
	};
protected:

	int						_iImgOrder;
};
#endif		/*!< BLEND_BASE_H */
