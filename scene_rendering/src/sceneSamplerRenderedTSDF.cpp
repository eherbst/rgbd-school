/*
 * sceneSamplerRenderedTSDF: a sampler for a single rendering of a Peter-Intel TSDF
 *
 * Evan Herbst
 * 11 / 6 / 13
 */

#include "scene_rendering/sceneSamplerRenderedTSDF.h"

sceneSamplerRenderedTSDF::sceneSamplerRenderedTSDF(const rgbd::CameraParams& cp, const std::shared_ptr<std::vector<float>>& p, const cv::Mat_<cv::Vec4b>& c, const std::shared_ptr<std::vector<float>>& n, const std::shared_ptr<std::vector<int>>& v)
: camParams(cp), pts(p), normals(n), cols(c), validity(v)
{}

uint64_t sceneSamplerRenderedTSDF::numSamples() const
{
	return validity->size();
}

/*
 * output arrays will be allocated if nec
 */
void sceneSamplerRenderedTSDF::getPerPixelInfo(const boost::multi_array<uint32_t, 2>& sampleIDsPerPixel, boost::multi_array<rgbd::eigen::Vector3f, 2>& expectedNormalsPerPixel, boost::multi_array<boost::array<uint8_t, 3>, 2>& expectedColsPerPixel) const
{
	expectedNormalsPerPixel.resize(boost::extents[sampleIDsPerPixel.shape()[0]][sampleIDsPerPixel.shape()[1]]);
	expectedColsPerPixel.resize(boost::extents[sampleIDsPerPixel.shape()[0]][sampleIDsPerPixel.shape()[1]]);
#if 1 //speed test 20131106 -- verdict: ~2x speedup
	const size_t n0 = sampleIDsPerPixel.shape()[0], n1 = sampleIDsPerPixel.shape()[1];
	const uint32_t* id = sampleIDsPerPixel.data();
	float* n = reinterpret_cast<float*>(expectedNormalsPerPixel.data());
	uint8_t* c = reinterpret_cast<uint8_t*>(expectedColsPerPixel.data());
	for(size_t i = 0; i < n0; i++)
		for(size_t j = 0; j < n1; j++, id++)
			if(*id > 0 && (*validity)[*id - 1])
			{
				for(int k = 0; k < 3; k++) *n++ = (*normals)[(*id - 1) * 4 + k];
				for(int k = 0; k < 3; k++) *c++ = cols((*id - 1) / cols.cols, (*id - 1) % cols.cols)[2 - k];
			}
#else //works
	for(size_t i = 0; i < sampleIDsPerPixel.shape()[0]; i++)
		for(size_t j = 0; j < sampleIDsPerPixel.shape()[1]; j++)
			if(sampleIDsPerPixel[i][j] > 0 && (*validity)[sampleIDsPerPixel[i][j] - 1])
			{
				for(int k = 0; k < 3; k++) expectedNormalsPerPixel[i][j][k] = (*normals)[(sampleIDsPerPixel[i][j] - 1) * 4 + k];
				for(int k = 0; k < 3; k++) expectedColsPerPixel[i][j][k] = cols((sampleIDsPerPixel[i][j] - 1) / cols.cols, (sampleIDsPerPixel[i][j] - 1) % cols.cols)[2 - k];
			}
#endif
}
void sceneSamplerRenderedTSDF::getSamplePts(std::vector<rgbd::eigen::Vector3f>& samplePts) const
{
	samplePts.resize(validity->size());
	const int* v = validity->data();
	float* x = reinterpret_cast<float*>(samplePts.data());
	for(size_t i = 0; i < validity->size(); i++, v++)
		if(*v)
			for(int k = 0; k < 3; k++) *x++ = (*pts)[i * 4 + k];
}
void sceneSamplerRenderedTSDF::getSampleNormals(std::vector<rgbd::eigen::Vector3f>& sampleNormals) const
{
	sampleNormals.resize(validity->size());
	const int* v = validity->data();
	float* n = reinterpret_cast<float*>(sampleNormals.data());
	for(size_t i = 0; i < validity->size(); i++, v++)
		if(*v)
			for(int k = 0; k < 3; k++) *n++ = (*normals)[i * 4 + k];
}
void sceneSamplerRenderedTSDF::getSampleCols(std::vector<boost::array<uint8_t, 3>>& sampleCols) const
{
	sampleCols.resize(validity->size());
	const int* v = validity->data();
	uint8_t* c = reinterpret_cast<uint8_t*>(sampleCols.data());
	for(size_t i = 0; i < validity->size(); i++, v++)
		if(*v)
			for(int k = 0; k < 3; k++) *c++ = cols(i / cols.cols, i % cols.cols)[k];
}
