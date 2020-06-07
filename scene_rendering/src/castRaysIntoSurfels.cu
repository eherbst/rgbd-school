/*
 * castRaysIntoSurfels: associate locations in an image plane with surfels in a cloud
 *
 * Evan Herbst
 * 7 / 9 / 13
 */

#include <stdint.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include "cuda_util/cudaUtils.h"

static const uint32_t blockSize1D = 32; //max cuda threads per block dimension; use 32 for Fermi architecture, 16 for Tesla

__global__ void remapDepthsFromUnitRangeKernel(const uint32_t* ids, float* depths, const uint32_t width, const uint32_t height, const float znear, const float zfar)
{
	const uint32_t i = blockIdx.y * blockSize1D + threadIdx.y, j = blockIdx.x * blockSize1D + threadIdx.x, l = i * width + j;
	if(ids[l] > 0) depths[l] = (znear * zfar / (znear - zfar)) / (depths[l] - zfar / (zfar - znear));
}

/*
 * edit depthsPBO to remap depths from [0, 1] to physical values
 */
void remapDepthsFromUnitRangeCUDA(const uint32_t width, const uint32_t height, const GLuint idsPBO, const GLuint depthsPBO, const float znear, const float zfar)
{
	CUDA_CALL(cudaGLRegisterBufferObject(idsPBO));
	CUDA_CALL(cudaGLRegisterBufferObject(depthsPBO));
	uint32_t* idPtr;
	float* depthPtr;
	CUDA_CALL(cudaGLMapBufferObject((void**)&idPtr, idsPBO));
	CUDA_CALL(cudaGLMapBufferObject((void**)&depthPtr, depthsPBO));

	const dim3 blockSize(blockSize1D, blockSize1D, 1);
	const dim3 numBlocks((uint32_t)ceil(width / blockSize.x), (uint32_t)ceil(height / blockSize.y), 1);
	remapDepthsFromUnitRangeKernel<<<numBlocks, blockSize>>>(idPtr, depthPtr, width, height, znear, zfar);

	CUDA_CALL(cudaGLUnmapBufferObject(idsPBO));
	CUDA_CALL(cudaGLUnmapBufferObject(depthsPBO));
	CUDA_CALL(cudaGLUnregisterBufferObject(idsPBO));
	CUDA_CALL(cudaGLUnregisterBufferObject(depthsPBO));
}
