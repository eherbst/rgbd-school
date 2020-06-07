
#include <memory>
#include <stdexcept>
#include "volume_modeler_compiled/cll.h" //opencl
#include "volume_modeler_compiled/volume_modeler.h"
namespace fs = boost::filesystem;

int main(int argc, char* argv[])
{
	ParamsVolumeModeler params_volume_modeler;
	ParamsCamera params_camera;
	ParamsFeatures params_features; //for point features
	ParamsGrid params_grid; //for the grid of patch volumes
	ParamsNormals params_normals;
	ParamsAlignment params_optimize;
	ParamsVolume params_volume;
	ParamsLoopClosure params_loop_closure;
	std::unique_ptr<CL> cl_ptr; //opencl
	boost::shared_ptr<OpenCLAllKernels> clKernels;
	std::shared_ptr<VolumeModeler> mapper;

	const OpenCLPlatformType platform_type = OPENCL_PLATFORM_NVIDIA; //: OPENCL_PLATFORM_INTEL;
	const OpenCLContextType context_type = OPENCL_CONTEXT_DEFAULT;// OPENCL_CONTEXT_CPU; // OPENCL_CONTEXT_GPU;
	cl_ptr.reset(new CL(platform_type, context_type));
	if(!cl_ptr->isInitialized()) throw new std::runtime_error("Failed to initialize OpenCL");

	const fs::path peterOpenclSrcPath = "/usr/local/proj/peter-intel-mapping/volume_modeler_compiled"; //TODO parameterize
	clKernels.reset(new OpenCLAllKernels(*cl_ptr, peterOpenclSrcPath));

	mapper.reset(new VolumeModeler(clKernels, params_volume_modeler, params_camera, params_volume, params_features, params_optimize, params_normals, params_grid, params_loop_closure));

	return 0;
}
