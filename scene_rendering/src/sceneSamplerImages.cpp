/*
 * sceneSamplerImages: take samples from a scene already rendered into images
 *
 * Evan Herbst
 * 1 / 15 / 14
 */

#include <algorithm>
#include "scene_rendering/sceneSamplerImages.h"

sceneSamplerImages::sceneSamplerImages(rgbdFrame& frame)
{
	const cv::Mat_<cv::Vec3b>& colorImg = frame.getColorImg();
	pixelCols.resize(boost::extents[colorImg.rows][colorImg.cols]);
	memcpy(pixelCols.data(), colorImg.ptr<uint8_t>(0), colorImg.rows * colorImg.cols * 3);
	//std::copy_n(colorImg.ptr<uint8_t>(0), colorImg.rows * colorImg.cols * 3, reinterpret_cast of pixelCols.data());
	const pcl::PointCloud<rgbd::pt>::ConstPtr cloud = frame.getOrganizedCloud();
	pixelPts.resize(boost::extents[cloud->height][cloud->width]);
	for(size_t i = 0; i < cloud->points.size(); i++) *(pixelPts.data() + i) = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(cloud->points[i]);
	const boost::multi_array<rgbd::eigen::Vector3f, 2>& normals = frame.getNormals();
	pixelNormals.resize(boost::extents[normals.shape()[0]][normals.shape()[1]]);
	std::copy_n(normals.data(), normals.num_elements(), pixelNormals.data());
}

uint64_t sceneSamplerImages::numSamples() const
{
	return pixelPts.shape()[0] * pixelPts.shape()[1];
}

/*
 * output arrays will be allocated if nec
 */
void sceneSamplerImages::getPerPixelInfo(const boost::multi_array<uint32_t, 2>& sampleIDsPerPixel, boost::multi_array<rgbd::eigen::Vector3f, 2>& expectedNormalsPerPixel, boost::multi_array<boost::array<uint8_t, 3>, 2>& expectedColsPerPixel) const
{
	//assume sample ids are the obvious: at pixels with valid depth, row-major pixel index; else 0
	expectedColsPerPixel.resize(boost::extents[pixelPts.shape()[0]][pixelPts.shape()[1]]);
	expectedNormalsPerPixel.resize(boost::extents[pixelPts.shape()[0]][pixelPts.shape()[1]]);
	expectedColsPerPixel = pixelCols;
	expectedNormalsPerPixel = pixelNormals;
}
void sceneSamplerImages::getSamplePts(std::vector<rgbd::eigen::Vector3f>& samplePts) const
{
	samplePts.resize(pixelPts.num_elements());
	std::copy_n(pixelPts.data(), pixelPts.num_elements(), samplePts.begin());
}
void sceneSamplerImages::getSampleNormals(std::vector<rgbd::eigen::Vector3f>& sampleNormals) const
{
	sampleNormals.resize(pixelNormals.num_elements());
	std::copy_n(pixelNormals.data(), pixelNormals.num_elements(), sampleNormals.begin());
}
void sceneSamplerImages::getSampleCols(std::vector<boost::array<uint8_t, 3>>& sampleCols) const
{
	sampleCols.resize(pixelCols.num_elements());
	std::copy_n(pixelCols.data(), pixelCols.num_elements(), sampleCols.begin());
}
