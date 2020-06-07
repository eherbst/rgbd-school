/*
 * sceneSamplerImages: take samples from a scene already rendered into images
 *
 * Evan Herbst
 * 1 / 15 / 14
 */

#ifndef EX_SCENE_SAMPLER_IMAGES_H
#define EX_SCENE_SAMPLER_IMAGES_H

#include <boost/multi_array.hpp>
#include "rgbd_frame_common/rgbdFrame.h"
#include "scene_rendering/sceneSampler.h"

class sceneSamplerImages: public sceneSampler
{
	public:

		sceneSamplerImages(rgbdFrame& frame);
		virtual ~sceneSamplerImages() {}

		virtual uint64_t numSamples() const;

		/*
		 * output arrays will be allocated if nec
		 */
		virtual void getPerPixelInfo(const boost::multi_array<uint32_t, 2>& sampleIDsPerPixel, boost::multi_array<rgbd::eigen::Vector3f, 2>& expectedNormalsPerPixel, boost::multi_array<boost::array<uint8_t, 3>, 2>& expectedColsPerPixel) const;
		virtual void getSamplePts(std::vector<rgbd::eigen::Vector3f>& samplePts) const;
		virtual void getSampleNormals(std::vector<rgbd::eigen::Vector3f>& sampleNormals) const;
		virtual void getSampleCols(std::vector<boost::array<uint8_t, 3>>& sampleCols) const;

	private:

		boost::multi_array<rgbd::eigen::Vector3f, 2> pixelPts;
		boost::multi_array<rgbd::eigen::Vector3f, 2> pixelNormals;
		boost::multi_array<boost::array<uint8_t, 3>, 2> pixelCols;
};

#endif //header
