/*
 * testModelerMemLeak: does VolumeModeler have a vram leak on destruction?
 *
 * Evan Herbst
 * 5 / 1 / 14
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include "opengl_util/openglContext.h"
#include "cuda_util/cudaUtils.h"
#include "volume_modeler_compiled/cll.h" //opencl
#include "volume_modeler_compiled/volume_modeler.h"
using std::cout;
using std::endl;

int main(int argc, char* argv[])
{
	openglContext ctx(480, 640);

	std::unique_ptr<CL> cl_ptr; //opencl
	boost::shared_ptr<OpenCLAllKernels> clKernels;

	const OpenCLPlatformType platform_type = OPENCL_PLATFORM_NVIDIA; //: OPENCL_PLATFORM_INTEL;
	const OpenCLContextType context_type = OPENCL_CONTEXT_DEFAULT;// OPENCL_CONTEXT_CPU; // OPENCL_CONTEXT_GPU;
	cl_ptr.reset(new CL(platform_type, context_type));
	if(!cl_ptr->isInitialized()) exit(-1);
	clKernels.reset(new OpenCLAllKernels(*cl_ptr, "/usr/local/proj/peter-intel-mapping/volume_modeler_compiled"));

	VolumeModelerAllParams modelerParams;
	modelerParams.volume_modeler.model_type = MODEL_SINGLE_VOLUME;
	std::shared_ptr<VolumeModeler> mapper1(new VolumeModeler(clKernels, modelerParams));
	std::shared_ptr<VolumeModeler> mapper2;

	ctx.acquire(); //needed for the nvidia ram functions (implemented as a gl extension)
	for(int i = 0; i < 100; i++)
	{
		cout << "iter " << i << ": vram total " << getNVIDIAVRAMTotal() << ", free " << getNVIDIAVRAMFree() << endl;
		mapper2.reset(mapper1->clone());//new VolumeModeler(clKernels, modelerParams));
		//mapper->load("/media/sdb/scene_matching/datasets/icra14A/maps/icra14A1.peterIntel");
		//mapper1->deallocateBuffers(); doesn't matter
		//std::this_thread::sleep_for(std::chrono::milliseconds(500)); doesn't matter
	}

	return 0;
}
