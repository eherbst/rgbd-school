/*
 * sceneSamplerPointCloud: take samples from a scene represented as a point cloud
 *
 * Evan Herbst
 * 2 / 29 / 12
 */

#ifndef EX_SCENE_SAMPLER_POINTCLOUD_H
#define EX_SCENE_SAMPLER_POINTCLOUD_H

#include <vector>
#include <pcl/point_cloud.h>
#include "pcl_rgbd/pointTypes.h"
#include "scene_rendering/sceneSampler.h"

class sceneSamplerPointCloud : public sceneSampler
{
	public:

		/*
		 * pre: c has normals set
		 */
		sceneSamplerPointCloud(const pcl::PointCloud<rgbd::pt>& c) : cloud(c) {}
		virtual ~sceneSamplerPointCloud() {}

		virtual uint64_t numSamples() const {return cloud.points.size();}

		/*
		 * output arrays will be allocated if nec
		 */
		virtual void getPerPixelInfo(const boost::multi_array<uint32_t, 2>& sampleIDsPerPixel, boost::multi_array<rgbd::eigen::Vector3f, 2>& expectedNormalsPerPixel, boost::multi_array<boost::array<uint8_t, 3>, 2>& expectedColsPerPixel) const;
		virtual void getSamplePts(std::vector<rgbd::eigen::Vector3f>& samplePts) const;
		virtual void getSampleNormals(std::vector<rgbd::eigen::Vector3f>& sampleNormals) const;
		virtual void getSampleCols(std::vector<boost::array<uint8_t, 3>>& sampleCols) const;

	private:

		const pcl::PointCloud<rgbd::pt>& cloud;
};

#endif //header
