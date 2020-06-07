/*
 * sceneSamplerRenderedTSDF: a sampler for a single rendering of a Peter-Intel TSDF
 *
 * Evan Herbst
 * 7 / 3 / 13
 */

#ifndef EX_SCENE_SAMPLER_RENDERED_TSDF_H
#define EX_SCENE_SAMPLER_RENDERED_TSDF_H

#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include "rgbd_util/CameraParams.h"
#include "scene_rendering/sceneSampler.h"

class sceneSamplerRenderedTSDF: public sceneSampler
{
	public:

		sceneSamplerRenderedTSDF(const rgbd::CameraParams& cp, const std::shared_ptr<std::vector<float>>& p, const cv::Mat_<cv::Vec4b>& c, const std::shared_ptr<std::vector<float>>& n, const std::shared_ptr<std::vector<int>>& v);
		virtual ~sceneSamplerRenderedTSDF() {}

		virtual uint64_t numSamples() const;

		/*
		 * output arrays will be allocated if nec
		 */
		virtual void getPerPixelInfo(const boost::multi_array<uint32_t, 2>& sampleIDsPerPixel, boost::multi_array<rgbd::eigen::Vector3f, 2>& expectedNormalsPerPixel, boost::multi_array<boost::array<uint8_t, 3>, 2>& expectedColsPerPixel) const;
		virtual void getSamplePts(std::vector<rgbd::eigen::Vector3f>& samplePts) const;
		virtual void getSampleNormals(std::vector<rgbd::eigen::Vector3f>& sampleNormals) const;
		virtual void getSampleCols(std::vector<boost::array<uint8_t, 3>>& sampleCols) const;

	private:

		const rgbd::CameraParams camParams;
		const std::shared_ptr<std::vector<float>> pts, normals; //4 per pixel
		const cv::Mat_<cv::Vec4b> cols; //bgra
		const std::shared_ptr<std::vector<int>> validity; //1 per pixel
};

#endif //header
