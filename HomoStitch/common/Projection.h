#ifndef PROJECT_H
#define PROJECT_H

#include <opencv2/core.hpp>

#include <cmath>

namespace pano {
	typedef cv::Point2f(*homo2proj_t)(const cv::Point3f&);

	typedef cv::Point3f(*proj2homo_t)(const cv::Point2f&);

	namespace flat {
		static inline cv::Point2f homo2proj(const cv::Point3f& homo) {
			return cv::Point2f(homo.x / homo.z, homo.y / homo.z);
		}

		// input & gradInput
		// given h & dh/dx, return dp/dx = dp/dh * dh/dx
		static inline cv::Point2f gradproj(const cv::Point3f& homo, const cv::Point3f& gradhomo) {
			double hz_inv = 1.0 / homo.z;
			double hz_sqr_inv = 1.0 / pow(homo.z, 2);
			return cv::Point2f{ float(gradhomo.x * hz_inv - gradhomo.z * homo.x * hz_sqr_inv),
									 float(gradhomo.y * hz_inv - gradhomo.z * homo.y * hz_sqr_inv) };
		}

		static inline cv::Point3f proj2homo(const cv::Point2f& proj) {
			return cv::Point3f(proj.x, proj.y, 1);
		}
	}

	namespace cylindrical {
		static inline cv::Point2f homo2proj(const cv::Point3f& homo) {
			return cv::Point2f(atan2(homo.x, homo.z),
				homo.y / (hypot(homo.x, homo.z)));
		}

		static inline cv::Point3f proj2homo(const cv::Point2f& proj) {
			return cv::Point3f(sin(proj.x), proj.y, cos(proj.x));
		}
	}

	namespace spherical {
		// not scale-invariant!
	// after mult by -1
	// y <-  -pi - y if (x<0) else pi - y
	// x <- x \pm pi
		static inline cv::Point2f homo2proj(const cv::Point3f& homo) {
			return cv::Point2f(atan2(homo.x, homo.z),
				atan2(homo.y, hypot(homo.x, homo.z)));
		}

		// input & gradInput
		// given h & dh/dx, return dp/dx = dp/dh * dh/dx
		static inline cv::Point2f gradproj(const cv::Point3f& homo, const cv::Point3f& gradhomo) {
			double h_xz = homo.x * homo.x + homo.z * homo.z,
				h_xz_r = sqrt(h_xz),
				h_xyz_inv = 1.0 / (h_xz + homo.y * homo.y),
				h_xz_inv = 1.0 / h_xz;
			return cv::Point2f{ float(gradhomo.x * homo.z * h_xz_inv - gradhomo.z * homo.x * h_xz_inv),
								float(-gradhomo.x * homo.x * homo.y * h_xyz_inv / h_xz_r
									 + gradhomo.y * h_xz_r * h_xyz_inv
									 - gradhomo.z * homo.y * homo.z * h_xyz_inv / h_xz_r) };
		}

		static inline cv::Point3f proj2homo(const cv::Point2f& proj) {
			return cv::Point3f(sin(proj.x), tan(proj.y), cos(proj.x));
		}
	}
}

#endif		/*!< PROJECT_H */
