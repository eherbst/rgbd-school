/*
 * sceneSamplerCUDARenderedTSDF: a sampler for a single rendering of a Peter-Intel TSDF, using cuda
 *
 * Evan Herbst
 * 7 / 8 / 13
 */

#ifndef EX_SCENE_SAMPLER_CUDA_RENDERED_TSDF_H
#define EX_SCENE_SAMPLER_CUDA_RENDERED_TSDF_H

#include <stdint.h>
#include <vector>
#include <thrust/device_vector.h>
#include "cuda_util/cudaUtils.h"

/*
 * map "sample" IDs to position, color, normal and whether the id is valid
 */
class cudaSceneSamplerRenderedTSDF
{
	public:

		cudaSceneSamplerRenderedTSDF(const uint32_t imgWidth, const uint32_t imgHeight, const std::vector<float>& p, const uchar4* c, const std::vector<float>& n, const std::vector<int32_t>& v);
		~cudaSceneSamplerRenderedTSDF() {}

		/*
		 * fill per-pixel buffers w/ col, normal, validity given a pixel-to-id map
		 */
		void getAttributesForIDs(const thrust::device_vector<uint32_t>& idImg, thrust::device_vector<uchar4>& colImg, thrust::device_vector<float3>& normalImg, thrust::device_vector<uint8_t>& validityImg) const;

	private:

		//indexed by sample id
		thrust::device_vector<uchar4> cudaCols;
		thrust::device_vector<float4> cudaNormals;
		thrust::device_vector<int32_t> cudaValidity;
};

#endif //header
