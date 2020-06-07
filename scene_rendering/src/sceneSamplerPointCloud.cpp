/*
 * sceneSamplerPointCloud: take samples from a scene represented as a point cloud
 *
 * Evan Herbst
 * 9 / 3 / 13
 */

#include <iostream>
#include "rgbd_util/assert.h"
#include "scene_rendering/sceneSamplerPointCloud.h"

/*
 * output arrays will be allocated if nec
 */
void sceneSamplerPointCloud::getPerPixelInfo(const boost::multi_array<uint32_t, 2>& sampleIDsPerPixel, boost::multi_array<rgbd::eigen::Vector3f, 2>& expectedNormalsPerPixel, boost::multi_array<boost::array<uint8_t, 3>, 2>& expectedColsPerPixel) const
{
	expectedNormalsPerPixel.resize(boost::extents[sampleIDsPerPixel.shape()[0]][sampleIDsPerPixel.shape()[1]]);
	expectedColsPerPixel.resize(boost::extents[sampleIDsPerPixel.shape()[0]][sampleIDsPerPixel.shape()[1]]);
	for(size_t i = 0; i < sampleIDsPerPixel.shape()[0]; i++)
		for(size_t j = 0; j < sampleIDsPerPixel.shape()[1]; j++)
			if(sampleIDsPerPixel[i][j] > 0)
			{
				expectedNormalsPerPixel[i][j] = rgbd::ptNormal2eigen<rgbd::eigen::Vector3f>(cloud.points[sampleIDsPerPixel[i][j] - 1]);
				expectedColsPerPixel[i][j] = rgbd::unpackRGB<uint8_t>(cloud.points[sampleIDsPerPixel[i][j] - 1].rgb);
			}
}
void sceneSamplerPointCloud::getSamplePts(std::vector<rgbd::eigen::Vector3f>& samplePts) const
{
	samplePts.resize(cloud.points.size());
	for(size_t i = 0; i < cloud.points.size(); i++) samplePts[i] = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(cloud.points[i]);
}
void sceneSamplerPointCloud::getSampleNormals(std::vector<rgbd::eigen::Vector3f>& sampleNormals) const
{
	sampleNormals.resize(cloud.points.size());
	for(size_t i = 0; i < cloud.points.size(); i++) sampleNormals[i] = rgbd::ptNormal2eigen<rgbd::eigen::Vector3f>(cloud.points[i]);
}
void sceneSamplerPointCloud::getSampleCols(std::vector<boost::array<uint8_t, 3>>& sampleCols) const
{
	sampleCols.resize(cloud.points.size());
	for(size_t i = 0; i < cloud.points.size(); i++) sampleCols[i] = rgbd::unpackRGB<uint8_t>(cloud.points[i].rgb);
}
