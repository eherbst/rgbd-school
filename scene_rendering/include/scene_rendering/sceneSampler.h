/*
 * sceneSampler: take samples from some map representation so we can do sample-based things like differencing
 *
 * Evan Herbst
 * 2 / 15 / 12
 */

#ifndef EX_SCENE_SAMPLER_H
#define EX_SCENE_SAMPLER_H

#include <cstdint>
#include <boost/multi_array.hpp>
#include <boost/array.hpp>
#include "rgbd_util/eigen/Geometry"

class sceneSampler
{
	public:

		virtual ~sceneSampler() {}

		virtual uint64_t numSamples() const = 0;

		/*
		 * output arrays will be allocated if nec
		 */
		virtual void getPerPixelInfo(const boost::multi_array<uint32_t, 2>& sampleIDsPerPixel, boost::multi_array<rgbd::eigen::Vector3f, 2>& expectedNormalsPerPixel, boost::multi_array<boost::array<uint8_t, 3>, 2>& expectedColsPerPixel) const = 0;
		virtual void getSamplePts(std::vector<rgbd::eigen::Vector3f>& samplePts) const = 0;
		virtual void getSampleNormals(std::vector<rgbd::eigen::Vector3f>& sampleNormals) const = 0;
		virtual void getSampleCols(std::vector<boost::array<uint8_t, 3>>& sampleCols) const = 0;
};

#endif //header
