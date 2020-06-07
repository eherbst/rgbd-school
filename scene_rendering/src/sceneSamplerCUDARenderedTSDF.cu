/*
 * sceneSamplerCUDARenderedTSDF: a sampler for a single rendering of a Peter-Intel TSDF, using cuda
 *
 * Evan Herbst
 * 7 / 8 / 13
 */

#include "scene_rendering/sceneSamplerCUDARenderedTSDF.h"

cudaSceneSamplerRenderedTSDF::cudaSceneSamplerRenderedTSDF(const uint32_t imgWidth, const uint32_t imgHeight, const std::vector<float>& p, const uchar4* c, const std::vector<float>& n, const std::vector<int32_t>& v)
{
	cudaCols.resize(imgHeight * imgWidth);
	cudaNormals.resize(imgHeight * imgWidth);
	cudaValidity.resize(imgHeight * imgWidth);
	thrust::copy(c, c + imgHeight * imgWidth, cudaCols.begin());
	thrust::copy(reinterpret_cast<const float4*>(n.data()), reinterpret_cast<const float4*>(n.data() + n.size()), cudaNormals.begin());
	thrust::copy(v.begin(), v.end(), cudaValidity.begin());
}

__global__ void getAttributesForIDsKernel(const uchar4* cudaCols, const float4* cudaNormals, const int32_t* cudaValidity, const uint32_t* ids, uchar4* colors, float3* normals, uint8_t* validity)
{
	const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t id = ids[i];
	colors[i] = cudaCols[id];
	normals[i].x = cudaNormals[id].x;
	normals[i].y = cudaNormals[id].y;
	normals[i].z = cudaNormals[id].z;
	validity[i] = cudaValidity[id];
}

/*
 * fill per-pixel buffers w/ col, normal, validity given a pixel-to-id map
 */
void cudaSceneSamplerRenderedTSDF::getAttributesForIDs(const thrust::device_vector<uint32_t>& idImg, thrust::device_vector<uchar4>& colImg, thrust::device_vector<float3>& normalImg, thrust::device_vector<uint8_t>& validityImg) const
{
	const dim3 blockSize(512/* TODO ? */, 1, 1), numBlocks((uint32_t)ceil(idImg.size() / blockSize.x), 1, 1);
	getAttributesForIDsKernel<<<numBlocks, blockSize>>>(cudaCols.data().get(), cudaNormals.data().get(), cudaValidity.data().get(), idImg.data().get(), colImg.data().get(), normalImg.data().get(), validityImg.data().get());
}
