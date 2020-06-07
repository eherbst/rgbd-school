/*
 * viewScoringRenderer: render views of a scene for scoring them as next best views
 *
 * Evan Herbst
 * 12 / 30 / 13
 */

#include <cassert>
#include <vector>
#include <cuda.h>
#include <thrust/device_vector.h>
#include "cuda_util/cudaUtils.h"

static const uint32_t maxBlockSize = 1024; //max cuda threads per block; TODO get from hardware?

/*
 * cuda textures are always file-static
 */
texture<uint8_t, 3/* dimensionality */, cudaReadModeElementType> colorTexRGBA;

/*
 * each block sums one row of one scene
 *
 * horizSums: row-major
 */
__global__ void scoreViewsHorizKernel(const size_t numScenesX, float* horizSums)
{
	/*
	 * all threads read into shared mem
	 */
	__shared__ uint8_t greenChannel[maxBlockSize]; //green channel is what we'll sum to get a score; size the array for the max possible block size but only use the actual block size
	//greenChannel[threadIdx.x] = cudaColorBufferRGBA[(gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x].y;
	greenChannel[threadIdx.x] = tex3D(colorTexRGBA, blockDim.x * blockIdx.x + threadIdx.x + .5f, blockDim.y * blockIdx.y + threadIdx.y + .5f, 1.5f);
	__syncthreads();

	/*
	 * first thread does summation
	 */
	if(threadIdx.x == 0)
	{
		uint64_t sum = 0;
		for(size_t i = 0; i < blockDim.x; i++) sum += greenChannel[i];
		horizSums[gridDim.x * blockIdx.y + blockIdx.x] = (float)sum;
	}
}

/*
 * each block sums one column of horiz scores, which is one scene
 *
 * horizSums: row-major
 */
__global__ void scoreViewsVertKernel(const size_t sceneSize, const float* horizSums, float* scores)
{
	/*
	 * all threads read into shared mem
	 */
	__shared__ float horizSumsCopy[maxBlockSize]; //size the array for the max possible block size but only use the actual block size
	horizSumsCopy[threadIdx.y] = horizSums[(gridDim.x * blockIdx.y + blockIdx.x) * blockDim.y + threadIdx.y];
	__syncthreads();

	/*
	 * first thread does summation
	 */
	if(threadIdx.y == 0)
	{
		float score = 0;
		for(size_t i = 0; i < blockDim.y; i++) score += horizSumsCopy[i];
		scores[gridDim.x * blockIdx.y + blockIdx.x] = score / sceneSize;
	}
}

/*
 * return scores (sum of green channel) for scenes in row-major order
 */
std::vector<float> scoreViewsInCUDA(const cudaArray* cudaColorBufferRGBA, const size_t bufWidth, const size_t bufHeight, const size_t numScenesX, const size_t numScenesY)
{
	/*
	 * set up texture
	 */

	colorTexRGBA.normalized = false;
	colorTexRGBA.filterMode = cudaFilterModePoint; //no interpolation
	for(uint32_t k = 0; k < 3; k++) colorTexRGBA.addressMode[k] = cudaAddressModeClamp;

//	const cudaExtent volumeSize = make_cudaExtent(imgWidth, imgHeight, imgDepth);
	const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();

	CUDA_CALL(cudaBindTextureToArray(colorTexRGBA, cudaColorBufferRGBA, channelDesc));

	/*
	 * sum elements
	 */

	std::vector<float> result(numScenesX * numScenesY);
	const size_t sceneWidth = bufWidth / numScenesX, sceneHeight = bufHeight / numScenesY;
	assert(sceneWidth <= maxBlockSize);
	assert(sceneHeight <= maxBlockSize);

	thrust::device_vector<float> devResultHoriz(numScenesX * bufHeight); //row-major single-scene row sums
{
	const dim3 blockSize(sceneWidth, 1, 1);
	const dim3 numBlocks(numScenesX, bufHeight, 1);
	scoreViewsHorizKernel<<<numBlocks, blockSize>>>(numScenesX, devResultHoriz.data().get());
	CUDA_CALL(cudaGetLastError());
}

	thrust::device_vector<float> devResult(numScenesX * numScenesY); //row-major single-scene scores
{
	const dim3 blockSize(1, sceneHeight, 1);
	const dim3 numBlocks(numScenesX, numScenesY, 1);
	scoreViewsVertKernel<<<numBlocks, blockSize>>>(sceneWidth * sceneHeight, devResultHoriz.data().get(), devResult.data().get());
	CUDA_CALL(cudaGetLastError());
}
	thrust::copy(devResult.begin(), devResult.end(), result.begin());
	return result;
}
