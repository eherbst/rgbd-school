/*
 * sceneSamplerSurfels: take samples from a scene represented as a surfel cloud
 *
 * Evan Herbst
 * 2 / 15 / 12
 */

#ifndef EX_SCENE_SAMPLER_SURFELS_H
#define EX_SCENE_SAMPLER_SURFELS_H

#include <cassert>
#include <vector>
#include "pcl_rgbd/pointTypes.h"
#include "scene_rendering/sceneSampler.h"

class sceneSamplerSurfels : public sceneSampler
{
	public:

		sceneSamplerSurfels(const pcl::PointCloud<rgbd::surfelPt>::Ptr& c, const std::vector<float>& pc) : cloud(c), princurv1(pc)
		{
			ASSERT_ALWAYS(princurv1.size() == cloud->points.size());
		}
		virtual ~sceneSamplerSurfels() {}

		virtual uint64_t numSamples() const {return cloud->points.size();}

		/*
		 * output arrays will be allocated if nec
		 */
		virtual void getPerPixelInfo(const boost::multi_array<uint32_t, 2>& sampleIDsPerPixel, boost::multi_array<rgbd::eigen::Vector3f, 2>& expectedNormalsPerPixel, boost::multi_array<boost::array<uint8_t, 3>, 2>& expectedColsPerPixel) const
		{
			expectedNormalsPerPixel.resize(boost::extents[sampleIDsPerPixel.shape()[0]][sampleIDsPerPixel.shape()[1]]);
			expectedColsPerPixel.resize(boost::extents[sampleIDsPerPixel.shape()[0]][sampleIDsPerPixel.shape()[1]]);
			for(size_t i = 0; i < sampleIDsPerPixel.shape()[0]; i++)
				for(size_t j = 0; j < sampleIDsPerPixel.shape()[1]; j++)
					if(sampleIDsPerPixel[i][j] > 0)
					{
						expectedNormalsPerPixel[i][j] = rgbd::ptNormal2eigen<rgbd::eigen::Vector3f>(cloud->points[sampleIDsPerPixel[i][j] - 1]);
						expectedColsPerPixel[i][j] = rgbd::unpackRGB<uint8_t>(cloud->points[sampleIDsPerPixel[i][j] - 1].rgb);
					}
		}
		virtual void getSamplePts(std::vector<rgbd::eigen::Vector3f>& samplePts) const
		{
			samplePts.resize(cloud->points.size());
			for(size_t i = 0; i < cloud->points.size(); i++) samplePts[i] = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(cloud->points[i]);
		}
		virtual void getSampleNormals(std::vector<rgbd::eigen::Vector3f>& sampleNormals) const
		{
			sampleNormals.resize(cloud->points.size());
			for(size_t i = 0; i < cloud->points.size(); i++) sampleNormals[i] = rgbd::ptNormal2eigen<rgbd::eigen::Vector3f>(cloud->points[i]);
		}
		virtual void getSampleCols(std::vector<boost::array<uint8_t, 3>>& sampleCols) const
		{
			sampleCols.resize(cloud->points.size());
			for(size_t i = 0; i < cloud->points.size(); i++) sampleCols[i] = rgbd::unpackRGB<uint8_t>(cloud->points[i].rgb);
		}

	private:

		const pcl::PointCloud<rgbd::surfelPt>::Ptr& cloud;
		const std::vector<float>& princurv1;
};

#endif //header
