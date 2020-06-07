/*
 * processHumanSegmDemo: process an obj segmentation demo given by a human (a main file for the icra14 submission)
 *
 * Evan Herbst
 * 8 / 29 / 13
 */

#include <cassert>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/gregorian/gregorian_types.hpp>
#include <cv_bridge/cv_bridge.h>
#include <graphcuts/GCoptimization.h>
#include "xforms/xforms.h"
#include "rgbd_util/primesensorUtils.h"
#include "rgbd_depthmaps/depthIO.h"
//#include "pcl_rgbd/cloudSearchTrees.h"
#include "pcl_rgbd/cloudTofroPLY.h"
#include "opengl_util/openglContext.h"
#include "evh_util/visualizationUtils.h"
#include "point_cloud_icp/registration/icp_utility.h"
#include "vrip_utils/meshTofroCloud.h"
#include "volume_modeler_compiled/cll.h" //opencl
#include "volume_modeler_compiled/volume_modeler.h"
#include "openni_utils/openni2FrameProvider.h"
#include "rgbd_bag_utils/rgbdBagReader2.h"
#include "rgbd_frame_common/staticDepthNoiseModeling.h"
#include "rgbd_frame_common/staticDepthNoiseModelingCUDA.h"
#include "rgbd_frame_common/rgbdFrame.h"
#include "peter_intel_mapping_utils/conversions.h"
#include "peter_intel_mapping_utils/io.h"
#include "scene_rendering/viewScoringRenderer.h"
#include "scene_rendering/sceneSamplerPointCloud.h"
#include "scene_rendering/sceneSamplerImages.h"
#include "scene_rendering/sceneSamplerRenderedTSDF.h"
#include "scene_rendering/pointCloudRenderer.h"
#include "scene_rendering/castRaysIntoSurfels.h"
#include "scene_differencing/sceneDifferencing.h"
#include "scene_differencing/sceneDifferencingMRF.h"
#include "scene_differencing/approxRGBDSensorNoiseModel.h"
#include "scene_differencing/visualization.h"
#include "optical_flow_utils/sceneFlowIO.h"
#include "rgbd_flow/rgbdFrameUtils.h" //downsizeFrame()
#include "rgbd_flow/runFlowTwoFrames.h"
#include "active_obj_modeling/diffingConnComps.h"
using std::cout;
using std::endl;
namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace gco;
using rgbd::eigen::Vector3f;
using rgbd::eigen::Affine3f;

void visualizeDepthStdevs(const boost::multi_array<float, 2>& stdevs, const std::string& filename)
{
	cv::Mat_<cv::Vec3b> stdevsImg(stdevs.shape()[0], stdevs.shape()[1]);
	for(size_t i = 0; i < stdevsImg.rows; i++)
		for(size_t j = 0; j < stdevsImg.cols; j++)
			if(stdevs[i][j] == 0) stdevsImg(i, j) = cv::Vec3b(255, 0, 0);
			else if(stdevs[i][j] < .003) stdevsImg(i, j) = cv::Vec3b(255, 255, 0);
			else if(stdevs[i][j] < .01) stdevsImg(i, j) = cv::Vec3b(0, 255, 0);
			else if(stdevs[i][j] < .03) stdevsImg(i, j) = cv::Vec3b(0, 255, 255);
			else if(stdevs[i][j] < .1) stdevsImg(i, j) = cv::Vec3b(0, 128, 255);
			else if(stdevs[i][j] < .3) stdevsImg(i, j) = cv::Vec3b(0, 0, 255);
			else stdevsImg(i, j) = cv::Vec3b(0, 0, 0);
	cv::imwrite(filename, stdevsImg);
}

struct diffingDetails
{
	//the rendering of the scene
	cv::Mat_<cv::Vec4b> renderedColsImg;
	cv::Mat_<float> renderedDepthsImg;

	//diffing internals
	cv::Mat_<float> dxImg;
};
void diffFrameVsPeterIntelMap(const rgbd::cameraSetup cams, const cv::Mat_<cv::Vec3b>& colorImg, const cv::Mat_<float>& depthImg, const pcl::PointCloud<rgbd::pt>::ConstPtr& organizedFrameCloudPtr,
	std::shared_ptr<VolumeModeler>& mapper, const Affine3f initialCamPoseWrtMap, const Affine3f framePoseWrtFrame0, rgbdSensorNoiseModel& noiseModel, sumLogprobsSeparate& frameSums,
	boost::optional<diffingDetails&> details)
{
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);

	diffingSingleFrameInfo scene2info;
	scene2info.sceneName = "s2";
	scene2info.frameDepth = depthImg;
	computeDepthMapStdevsPointwise(depthImg, scene2info.depthStdevs, noiseModel.getObsDepthSigma0(), 2/* # pixels near depth edges to increase uncertainty for; TODO ? */, true/* multithread */);
//	visualizeDepthStdevs(scene2info.depthStdevs, "frameStdevs.png");
	scene2info.frameImg = colorImg;
	scene2info.frameNormalsCloud = organizedFrameCloudPtr; //TODO use a multiarray for diffing speed
	scene2info.framePoseWrtMap = Affine3f::Identity();

	const Affine3f map2PoseWrtMap1 = Affine3f::Identity();
	mapper->render(initialCamPoseWrtMap * framePoseWrtFrame0);
	//args to getLastRender() do need to be preallocated
	cv::Mat prevRenderedCols(camParams.yRes, camParams.xRes, cv::DataType<cv::Vec4b>::type); //bgra
	std::shared_ptr<std::vector<float>> prevRenderedPts(new std::vector<float>(camParams.yRes * camParams.xRes * 4));
	std::shared_ptr<std::vector<float>> prevRenderedNormals(new std::vector<float>(camParams.yRes * camParams.xRes * 4));
	std::shared_ptr<std::vector<int>> prevRenderingValidity(new std::vector<int>(camParams.yRes * camParams.xRes)); //iff 0 at a pixel, pts, normals and maybe colors are nan
	mapper->getLastRender(prevRenderedCols, *prevRenderedPts, *prevRenderedNormals, *prevRenderingValidity);
	std::shared_ptr<sceneSamplerRenderedTSDF> scene1sampler(new sceneSamplerRenderedTSDF(camParams, prevRenderedPts, prevRenderedCols, prevRenderedNormals, prevRenderingValidity));
	const projectSamplesIntoCameraPerPixelFunc sampleProjectionFunc = [&prevRenderedPts,&prevRenderingValidity](const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camXform, boost::multi_array<uint32_t, 2>& sampleIDs, boost::multi_array<float, 2>& sampleDepths, cv::Mat_<float>& sampleDepthStdevs, bool& stdevsSet)
		{
			for(size_t i = 0, l = 0; i < camParams.yRes; i++)
			{
				for(size_t j = 0; j < camParams.xRes; j++, l++)
				{
					sampleIDs[i][j] = (*prevRenderingValidity)[l] ? (l + 1) : 0; //0 in this map is a flag for no sample
					sampleDepths[i][j] = (*prevRenderingValidity)[l] ? (*prevRenderedPts)[l * 4 + 2] : -1;
				}
			}

			const cv::Mat_<float> depthsMat(camParams.yRes, camParams.xRes, sampleDepths.data()); //won't try to deallocate when dies
			boost::multi_array<float, 2> sampleDepthStdevsMA(boost::extents[camParams.yRes][camParams.xRes]);
#if 0 //run on the gpu: ~6x speedup
			computeDepthMapLocalStdevsCUDA(depthsMat, 3/* nbrhood */, sampleDepthStdevsMA, 0, camParams, true/* multithread */);
#if 0 //for comparison of values (20131105 they're similar)
			boost::multi_array<float, 2> sampleDepthStdevsMA2(boost::extents[camParams.yRes][camParams.xRes]);
			computeDepthMapLocalStdevs(depthsMat, 3/* nbrhood */, sampleDepthStdevsMA2, 0, camParams, true/* multithread */);
			for(size_t i = 0; i < camParams.yRes; i += 10)
				for(size_t j = 0; j < camParams.xRes; j += 10)
					cout << sampleDepthStdevsMA[i][j] << ',' << sampleDepthStdevsMA2[i][j] << "   ";
			cout << endl;
#endif
#else
			computeDepthMapLocalStdevs(depthsMat, 3/* nbrhood */, sampleDepthStdevsMA, 0, camParams, true/* multithread */);
#endif
//			visualizeDepthStdevs(sampleDepthStdevsMA, "mapStdevs.png");
			std::copy(sampleDepthStdevsMA.data(), sampleDepthStdevsMA.data() + sampleDepthStdevsMA.num_elements(), reinterpret_cast<float*>(sampleDepthStdevs.data));
			stdevsSet = true;
		};
	noiseModel.cachePerPointInfo(*scene1sampler);

	sceneDifferencer differ(cams);
	differ.setPrevScene(scene1sampler);
	differ.setCurFrame(scene2info, map2PoseWrtMap1);
	differ.projectSceneIntoFrame(sampleProjectionFunc);
	singleFrameDifferencingParams diffingParams;

	//experimental stuff
	if(details)
	{
		(*details).renderedColsImg = prevRenderedCols.clone();
		for(size_t i = 0, l = 0; i < camParams.yRes; i++)
			for(size_t j = 0; j < camParams.xRes; j++, l++)
			{
				if((*prevRenderingValidity)[l]) (*details).renderedDepthsImg(i, j) = (*prevRenderedPts)[l * 4 + 2];
				else (*details).renderedDepthsImg(i, j) = 0;

				if((*prevRenderingValidity)[l] && depthImg(i, j) > 0)
					(*details).dxImg(i, j) = (depthImg(i, j) - (*prevRenderedPts)[l * 4 + 2]) / scene2info.depthStdevs[i][j];
				else
					(*details).dxImg(i, j) = -FLT_MAX; //flag for invalid
			}
	}

	/*
	 * output wrt new frame
	 */
	diffingParams.differenceFrameWrtCloud = false;
	diffingParams.outputWrtCloud = false;
	assert(frameSums.size() == camParams.xRes * camParams.yRes);
	differ.runDifferencingAfterProjection(noiseModel, frameSums, diffingParams);
	const float posWeight = .4, colWeight = 0/*.4*//*0*/, normalWeight = 0; //TODO ?
	frameSums.weightComponents(posWeight, colWeight, normalWeight);
}

int main(int argc, char* argv[])
{
	po::options_description desc("Options");
	desc.add_options()
		("inpath,i", po::value<fs::path>(), "path to a bag or oni, or if == 'ONI', will use real-time openni2")
//		("noise-model-file,g", po::value<fs::path>(), "path to a noise model paramfile")
		("read-intermediate,v", po::value<bool>()->default_value(false), "whether to read intermediate results (reconstruction results and demo start/end frame info) from disk")
		("outdir,o", po::value<fs::path>(), "will be created if nec")
		;
	po::variables_map vars;
	po::store(po::command_line_parser(argc, argv).options(desc).run(), vars);
	po::notify(vars);

	const fs::path outdir(vars["outdir"].as<fs::path>());
	fs::create_directories(outdir);

	const bool readVideoResultsFromDisk = vars["read-intermediate"].as<bool>();

	srand((unsigned int)time(NULL));

	rgbd::cameraSetup cams;
	cams.cam = rgbd::KINECT_640_DEFAULT;
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);

	/****************************************************************************************************************************************
	 * init diffing
	 */

	//for picking pixels to add to the bkgnd map
	std::shared_ptr<approxRGBDSensorNoiseModel> noiseModel(new approxRGBDSensorNoiseModel(.0001/* bias weight */, .008/* sigma_d0 */, .3/* col hit sigma */, 1.37/* max normal ang */, 40/* kappa normal hit */, 1e-7/* kappa normal miss */));

	//large sigma
	std::shared_ptr<rgbdSensorNoiseModel> noiseModelForStartEndDiffing(new approxRGBDSensorNoiseModel(.0001/* bias weight */, .03/* sigma_d0 */, .3/* col hit sigma */, 1.37/* max normal ang */, 40/* kappa normal hit */, 1e-7/* kappa normal miss */));

	std::shared_ptr<rgbdSensorNoiseModel> noiseModelForMofitting(new approxRGBDSensorNoiseModel(.0001/* bias weight */, .025/* sigma_d0 */, .3/* col hit sigma */, 1.37/* max normal ang */, 40/* kappa normal hit */, 1e-7/* kappa normal miss */));

	/****************************************************************************************************************************************
	 * init input source
	 */

	const fs::path bagFilepath = vars["inpath"].as<fs::path>();
	const bool useOpenni2 = (bagFilepath.string() == "ONI");
	std::shared_ptr<openni2FrameProvider> openniReader;
	std::shared_ptr<rgbd::rgbdBagReader2> bagReader;
	boost::posix_time::ptime frameTime(boost::gregorian::date(2000, 1, 1)); //only used if reading live from openni
	if(useOpenni2)
	{
		FrameProviderOpenni2Params readerParams;
		openniReader.reset(new openni2FrameProvider(readerParams));
	}
	else
	{
		bagReader.reset(new rgbd::rgbdBagReader2(bagFilepath, 0, 1000000, 0/* frameskip */, 0/* num prev frames to keep */));
	}

	/****************************************************************************************************************************************
	 * init the bkgnd map
	 */

	VolumeModelerAllParams modelerParams;
	std::unique_ptr<CL> cl_ptr; //opencl
	boost::shared_ptr<OpenCLAllKernels> clKernels;
	std::shared_ptr<VolumeModeler> bkgndMapper, demoBkgndMapper;

	const OpenCLPlatformType platform_type = OPENCL_PLATFORM_NVIDIA; //: OPENCL_PLATFORM_INTEL;
	const OpenCLContextType context_type = OPENCL_CONTEXT_DEFAULT;// OPENCL_CONTEXT_CPU; // OPENCL_CONTEXT_GPU;
	cl_ptr.reset(new CL(platform_type, context_type));
	if(!cl_ptr->isInitialized()) throw new std::runtime_error("Failed to initialize OpenCL");

	const fs::path peterOpenclSrcPath = "/usr/local/proj/peter-intel-mapping/volume_modeler_compiled"; //TODO parameterize
	clKernels.reset(new OpenCLAllKernels(*cl_ptr, peterOpenclSrcPath));

//	modelerParams.volume.cell_size = .005; //.005 takes forever in grid mode with any decent-sized video--unusable; TODO ?
	modelerParams.volume_modeler.model_type = MODEL_GRID;//MODEL_SINGLE_VOLUME;//MODEL_GRID;//MODEL_SINGLE_VOLUME;
	modelerParams.volume_modeler.use_features = false;
	modelerParams.volume_modeler.verbose = false;
	modelerParams.camera.focal = rgbd::eigen::Vector2f(camParams.focalLength, camParams.focalLength);
	modelerParams.camera.center = rgbd::eigen::Vector2f(camParams.centerX, camParams.centerY);
	modelerParams.grid.max_mb_gpu = 500; //reduce if running out of gpu memory; only relevant if using a grid
	bkgndMapper.reset(new VolumeModeler(clKernels, modelerParams));

	/****************************************************************************************************************************************
	 * build bkgnd map
	 */

	const float probMovedThreshold = .51; //TODO ?

	rgbd::eigen::Affine3f initialCamPoseWrtMap, initialCamPoseWrtDemoMap;
	std::vector<Affine3f> framePosesWrtFrame0;
	rgbdFrame startFrame, endFrame; //start and end of demo
	size_t demoStartFrameIndex, demoEndFrameIndex = std::numeric_limits<size_t>::max();
	if(readVideoResultsFromDisk)
	{

	{
		ifstream infile((outdir / "archive.txt").string().c_str());
		ASSERT_ALWAYS(infile);
		infile >> demoStartFrameIndex >> demoEndFrameIndex;
	}

		startFrame.getColorImgRef() = cv::imread((outdir / "demoStartFrameColorImg.png").string());
		rgbd::readDepthMapValuesImg(outdir / "demoStartFrameDepthImg.png", startFrame.getDepthImgRef());
		endFrame.getColorImgRef() = cv::imread((outdir / "demoEndFrameColorImg.png").string());
		rgbd::readDepthMapValuesImg(outdir / "demoEndFrameDepthImg.png", endFrame.getDepthImgRef());

		bkgndMapper->load(outdir / "bkgndMap");
	{
		const std::vector<rgbd::eigen::Affine3f> tmpCamPoses = xf::readTransformsTextFileNoTimestamps(outdir / "bkgndMap" / "mapOffsetXform.dat");
		ASSERT_ALWAYS(tmpCamPoses.size() == 1);
		initialCamPoseWrtMap = tmpCamPoses[0];
	}
		framePosesWrtFrame0 = xf::readTransformsTextFileNoTimestamps(outdir / "bkgndMap" / "frameXforms.dat");

		demoBkgndMapper.reset(new VolumeModeler(clKernels, modelerParams));
		demoBkgndMapper->load(outdir / "demoBkgndMap");
	{
		const std::vector<rgbd::eigen::Affine3f> tmpCamPoses = xf::readTransformsTextFileNoTimestamps(outdir / "demoBkgndMap" / "mapOffsetXform.dat");
		ASSERT_ALWAYS(tmpCamPoses.size() == 1);
		initialCamPoseWrtDemoMap = tmpCamPoses[0];
	}

	}
	else
	{

	rgbdFrame curFrame;
	enum class demoState : uint8_t {NO_DEMO, DEMO};
	demoState curDemoState = demoState::NO_DEMO;
	bool demoOver = false;
	const size_t numFramesToAggregateForDemoEndCheck = 50; //TODO ?
	std::deque<size_t> numMovedPtsByFrame; //for the last numFramesToAggregate frames
	size_t maxMovedPts = 0; //in a single frame during the demo
	size_t frameIndex = 0;
	while(!demoOver)
	{
		/*
		 * get frame
		 */
		cout << "reading frame " << frameIndex << endl;
		if(useOpenni2)
		{
			frameTime += boost::posix_time::seconds(1); //TODO ?
			if(!openniReader->getNextFrame(curFrame.getColorImgRef(), curFrame.getDepthImgRef(), frameTime)) break;
		}
		else
		{
			if(!bagReader->readOneFrame()) break;
			const sensor_msgs::ImageConstPtr img = bagReader->getLastUncompressedImgPtr();
			const cv_bridge::CvImageConstPtr ciMsg = cv_bridge::toCvShare(img, "bgr8");
			curFrame.getColorImgRef() = ciMsg->image.clone();
			curFrame.getDepthImgRef() = bagReader->getLastDepthImg().clone();
		}

		/*
		 * set normals efficiently
		 *
		 * TODO do we need more accurate normals in order for them to be useful for diffing?
		 */
	{
		std::vector<float> frameNormals;
		Frame frameToAdd(*cl_ptr);
		frameToAdd.mat_color = curFrame.getColorImg();
		frameToAdd.mat_depth = curFrame.getDepthImg();
		bkgndMapper->getNormals(frameToAdd, frameNormals); //doesn't matter what mapper we use; this won't change anything in the map
		boost::multi_array<rgbd::eigen::Vector3f, 2>& normalsImg = curFrame.getNormalsRef();
		normalsImg.resize(boost::extents[camParams.yRes][camParams.xRes]);
		for(size_t i = 0, l = 0; i < camParams.yRes; i++)
			for(size_t j = 0; j < camParams.xRes; j++, l++)
			{
				if(std::isnan(frameNormals[l * 4 + 0]))
					for(size_t k = 0; k < 3; k++) normalsImg[i][j][k] = 0;
				else
					for(size_t k = 0; k < 3; k++) normalsImg[i][j][k] = frameNormals[l * 4 + k];
			}
	}

		/*
		 * align to the bkgnd map
		 */
		Frame frame(*cl_ptr);
		frame.mat_color = curFrame.getColorImg(); //we won't edit this
		frame.mat_depth = curFrame.getDepthImg(); //we won't edit this
		if(frameIndex == 0)
		{
			//initialize the bkgnd map
			bkgndMapper->alignFrame(frame, initialCamPoseWrtMap); //need to do this at least once rather than just addFrame()ing so we get things more or less centered in the volume
			bkgndMapper->addFrame(frame, initialCamPoseWrtMap);
			framePosesWrtFrame0.push_back(Affine3f::Identity());
		}
		else
		{
			switch(curDemoState)
			{
				case demoState::NO_DEMO:
				{
					/*
					 * detect the start of a demo: diff wrt the cur bkgnd map
					 */
					sumLogprobsSeparate frameSums(camParams.xRes * camParams.yRes);
				{
					diffFrameVsPeterIntelMap(cams, curFrame.getColorImg(), curFrame.getDepthImg(), curFrame.getOrganizedCloud(), bkgndMapper, initialCamPoseWrtMap, framePosesWrtFrame0[frameIndex - 1], *noiseModelForStartEndDiffing, frameSums, boost::none);
		#if 0
				{
					const cv::Mat diffingWrtNewSceneVisImg = visualizeLogProbsWrtFrame(camParams, frameSums.evidenceMoved(), frameSums.evidenceNotMoved(), true/* multithread */);
					cv::imwrite((outdir / (boost::format("diffPreA%1%.png") % frameIndex).str()).string(), diffingWrtNewSceneVisImg);
				}
		#endif
					const rgbd::eigen::VectorXd movedProbs = frameSums.movedProbs();
		#if 1
				{
					uint64_t numMovedPixels = 0;
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
						for(size_t j = 0; j < camParams.xRes; j++, l++)
							if(movedProbs[l] > probMovedThreshold)
								numMovedPixels++;
					cout << numMovedPixels << " moved pixels" << endl;
				}
		#endif

					/*
					 * don't use obviously non-matching pixels for alignment -- this does improve foreground segmentation
					 */
					Frame frameToAdd(*cl_ptr);
					frameToAdd.mat_color = curFrame.getColorImg();
					frameToAdd.mat_depth = curFrame.getDepthImg().clone();
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
						for(size_t j = 0; j < camParams.xRes; j++, l++)
							if(movedProbs[l] > probMovedThreshold)
							{
								frameToAdd.mat_depth.at<float>(i, j) = 0;
							}

					Affine3f framePoseWrtMap;
					bkgndMapper->alignFrame(frameToAdd, framePoseWrtMap);
					framePosesWrtFrame0.push_back(initialCamPoseWrtMap.inverse() * framePoseWrtMap);

					/*
					 * re-diff w/ a new xform
					 */
					frameSums.init(camParams.xRes * camParams.yRes);
					diffFrameVsPeterIntelMap(cams, curFrame.getColorImg(), curFrame.getDepthImg(), curFrame.getOrganizedCloud(), bkgndMapper, initialCamPoseWrtMap, framePosesWrtFrame0[frameIndex], *noiseModelForStartEndDiffing, frameSums, boost::none);
#if 0
				{
					const cv::Mat diffingWrtNewSceneVisImg = visualizeLogProbsWrtFrame(camParams, frameSums.evidenceMoved(), frameSums.evidenceNotMoved(), true/* multithread */);
					cv::imwrite((outdir / (boost::format("diffPreB%1%.png") % frameIndex).str()).string(), diffingWrtNewSceneVisImg);
				}
#endif
				}

					/*
					 * find conn comps of moved pixels
					 */
					const rgbd::eigen::VectorXd movedProbs = frameSums.movedProbs();
					const std::unordered_map<size_t, std::vector<size_t>> compsByRep = std::move(findDiffingConnComps(camParams, curFrame.getDepthImg(), movedProbs, probMovedThreshold));

					cout << "comp sizes:";
					for(const auto& c : compsByRep)
					{
						cout << ' ' << c.second.size();
						if(c.second.size() > 1000/* TODO ? */) //for each large comp
						{
							//the comp is a moving obj (should be the user's arm); start the demo
							curDemoState = demoState::DEMO;
							demoStartFrameIndex = frameIndex;
							startFrame.getColorImgRef() = curFrame.getColorImg().clone();
							startFrame.getDepthImgRef() = curFrame.getDepthImg().clone();
							initialCamPoseWrtDemoMap = initialCamPoseWrtMap;
							demoBkgndMapper.reset(bkgndMapper->clone());
							cout << "demo start: frame " << frameIndex << endl;
							break;
						}
					}
					cout << endl;

					if(curDemoState == demoState::NO_DEMO) //if we didn't detect a demo starting (TODO right now we skip doing anything with this frame; can we avoid that?)
					{
						bkgndMapper->addFrame(frame, initialCamPoseWrtMap * framePosesWrtFrame0[frameIndex]);
					}

					break;
				}
				case demoState::DEMO:
				{
					/*
					 * diff wrt the cur bkgnd map using the prev frame's pose
					 */
					sumLogprobsSeparate frameSums(camParams.xRes * camParams.yRes);

					diffFrameVsPeterIntelMap(cams, curFrame.getColorImg(), curFrame.getDepthImg(), curFrame.getOrganizedCloud(), demoBkgndMapper, initialCamPoseWrtDemoMap, framePosesWrtFrame0[frameIndex - 1], *noiseModel, frameSums, boost::none);
		#if 1
				{
					const cv::Mat diffingWrtNewSceneVisImg = visualizeLogProbsWrtFrame(camParams, frameSums.evidenceMoved(), frameSums.evidenceNotMoved(), true/* multithread */);
					cv::imwrite((outdir / (boost::format("diffA%1%.png") % frameIndex).str()).string(), diffingWrtNewSceneVisImg);
				}
		#endif
					const rgbd::eigen::VectorXd movedProbs = frameSums.movedProbs();
				{
					uint64_t numMovedPixels = 0;
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
						for(size_t j = 0; j < camParams.xRes; j++, l++)
							if(movedProbs[l] > probMovedThreshold)
								numMovedPixels++;
					numMovedPtsByFrame.push_back(numMovedPixels);
					if(numMovedPtsByFrame.size() > numFramesToAggregateForDemoEndCheck) numMovedPtsByFrame.pop_front();
					if(numMovedPixels > maxMovedPts) maxMovedPts = numMovedPixels;
					cout << numMovedPixels << " moved pixels" << endl;
				}

					/*
					 * don't use obviously non-matching pixels for alignment -- this does improve foreground segmentation
					 */
					Frame frameToAdd(*cl_ptr);
					frameToAdd.mat_color = curFrame.getColorImg();
					frameToAdd.mat_depth = curFrame.getDepthImg().clone();
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
						for(size_t j = 0; j < camParams.xRes; j++, l++)
							if(movedProbs[l] > probMovedThreshold)
							{
								frameToAdd.mat_depth.at<float>(i, j) = 0;
							}

					Affine3f tmpPose;
					demoBkgndMapper->alignFrame(frameToAdd, tmpPose);
					framePosesWrtFrame0.push_back(initialCamPoseWrtDemoMap.inverse() * tmpPose);

					/*
					 * re-diff w/ a new xform
					 */
					diffingDetails details;
					details.dxImg.create(camParams.yRes, camParams.xRes);
					details.renderedColsImg.create(camParams.yRes, camParams.xRes);
					details.renderedDepthsImg.create(camParams.yRes, camParams.xRes);
					frameSums.init(camParams.xRes * camParams.yRes);
					diffFrameVsPeterIntelMap(cams, curFrame.getColorImg(), curFrame.getDepthImg(), curFrame.getOrganizedCloud(), demoBkgndMapper, initialCamPoseWrtDemoMap, framePosesWrtFrame0[frameIndex], *noiseModel, frameSums, details);
#if 1
				{
					const cv::Mat diffingWrtNewSceneVisImg = visualizeLogProbsWrtFrame(camParams, frameSums.evidenceMoved(), frameSums.evidenceNotMoved(), true/* multithread */);
					cv::imwrite((outdir / (boost::format("diffB%1%.png") % frameIndex).str()).string(), diffingWrtNewSceneVisImg);

					cv::imwrite((outdir / (boost::format("bkgndCols%1%.png") % frameIndex).str()).string(), details.renderedColsImg);
				}
#endif

					/*
					 * decide which pixels from the current frame we'll add to the bkgnd map
					 */

					const rgbd::eigen::VectorXd evidenceSum = frameSums.evidenceSum();
					const float gamma = 0; //multiplier for uncertain-state term; possibly unused
					const float smoothnessWeight = 1e1; //TODO ?
					const float movedPrior = .495; //make it so pixels without any info will default to not-moved
					const std::vector<mrfPointState> diffingMRFResult = std::move(runSceneDifferencingGraphCutsSingleFrame(curFrame, evidenceSum, frameSums.evidenceMagnitude(), gamma, smoothnessWeight, movedPrior));
#if 1
					cv::Mat_<cv::Vec3b> mrfImg(camParams.yRes, camParams.xRes);
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
						for(size_t j = 0; j < camParams.xRes; j++, l++)
						{
							switch(diffingMRFResult[l])
							{
								case mrfPointState::MOVED:
									mrfImg(i, j) = cv::Vec3b(0, 255, 255);
									break;
								case mrfPointState::NOT_MOVED:
									mrfImg(i, j) = cv::Vec3b(0, 0, 255);
									break;
								case mrfPointState::INVISIBLE:
									mrfImg(i, j) = cv::Vec3b(0, 0, 0);
									break;
							}
						}
					cv::imwrite((outdir / (boost::format("mrf%1%.png") % frameIndex).str()).string(), mrfImg);
#endif

					/*
					 * add to the bkgnd map
					 */
					Frame bkgndFrame(*cl_ptr);
					bkgndFrame.mat_color = curFrame.getColorImg().clone(); //we'll edit this
					bkgndFrame.mat_depth = curFrame.getDepthImg().clone(); //we'll edit this
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
						for(size_t j = 0; j < camParams.xRes; j++, l++)
						{
							//don't add pixels deemed to have changed
#if 1 //use the mrf result -- 20140116 does improve things a lot most of the time
							if(diffingMRFResult[l] == mrfPointState::MOVED)
								bkgndFrame.mat_depth.at<float>(i, j) = 0;
#else //don't use the mrf result
							if(evidenceSum[l] > 0)
								bkgndFrame.mat_depth.at<float>(i, j) = 0;
#endif
#if 1
							if(details.dxImg(i, j) != -FLT_MAX && details.dxImg(i, j) < 0) //if the bkgnd map surface is in front of what we see now, improve the bkgnd map
							{
								bkgndFrame.mat_depth.at<float>(i, j) = details.renderedDepthsImg(i, j);
								bkgndFrame.mat_color.at<cv::Vec3b>(i, j) = cv::Vec3b(details.renderedColsImg(i, j)[0], details.renderedColsImg(i, j)[1], details.renderedColsImg(i, j)[2]);
							}
#endif
#if 0
							//don't add things that changed wrt the start scene
							if(movedProbs[l] > .52/* TODO ? */)
								bkgndFrame.mat_depth.at<float>(i, j) = 0;
#endif
#if 0
							//don't add things that agree with the current bkgnd map (add only new and occluded bits)
							if(fabs(depthMovedProbs[l] - .5) > .01/* TODO ? */)
								bkgndFrame.mat_depth.at<float>(i, j) = 0;
#endif
						}
					/*
					 * with a max weight of 1.0 you can see that the frame-to-map alignments aren't that great--you get noticeable drift over hundreds of frames
					 *
					 * enforcing the max weight means you can overwrite early frames of movable object using info from later frames where you actually do see the bkgnd
					 */
					demoBkgndMapper->setMaxWeightInVolume(1.0/*3.0*/); //3.0 makes the map look somewhat smooth on pickupLunchboxWrtBkgndMotion1; 1.0 does not
					demoBkgndMapper->addFrame(bkgndFrame, initialCamPoseWrtDemoMap * framePosesWrtFrame0[frameIndex]);

					/*
					 * determine when the demo is over: over when # high-p(m) pts stabilizes at a large number
					 */
					float sum = 0, sqrSum = 0;
					for(size_t n : numMovedPtsByFrame)
					{
						sum += n;
						sqrSum += n * n;
					}
					const size_t count = numMovedPtsByFrame.size();
					const float mean = sum / count;
					const float stdev = sqrt(sqrSum / count - sqr(mean));
					cout << "demo end check: mean " << mean << ", stdev " << stdev << endl;
					if(count >= numFramesToAggregateForDemoEndCheck && mean > .4/* TODO ? */ * maxMovedPts && stdev < 20/* TODO ? */)
					{
						demoOver = true;
						demoEndFrameIndex = frameIndex;
						cout << "demo end at frame " << demoEndFrameIndex << endl;
					}

					break;
				}
			}
		}
		frameIndex++;
	}
	if(demoEndFrameIndex == std::numeric_limits<size_t>::max()) demoEndFrameIndex = frameIndex - 1;
	endFrame.getColorImgRef() = curFrame.getColorImg().clone();
	endFrame.getDepthImgRef() = curFrame.getDepthImg().clone();

		/*
		 * write everything we'll need later to disk
		 */
	{
		ofstream outfile((outdir / "archive.txt").string().c_str());
		ASSERT_ALWAYS(outfile);
		outfile << demoStartFrameIndex << ' ' << demoEndFrameIndex << endl;
	}

		cv::imwrite((outdir / "demoStartFrameColorImg.png").string(), startFrame.getColorImg());
		rgbd::writeDepthMapValuesImg(startFrame.getDepthImg(), outdir / "demoStartFrameDepthImg.png");
		cv::imwrite((outdir / "demoEndFrameColorImg.png").string(), endFrame.getColorImg());
		rgbd::writeDepthMapValuesImg(endFrame.getDepthImg(), outdir / "demoEndFrameDepthImg.png");

		writePeterIntelMap(*bkgndMapper, initialCamPoseWrtMap, outdir / "bkgndMap", true/* write meshes */, true/* write surfels */);
		xf::writeTransformsTextFile(framePosesWrtFrame0, outdir / "bkgndMap" / "frameXforms.dat");
		writePeterIntelMap(*demoBkgndMapper, initialCamPoseWrtDemoMap, outdir / "demoBkgndMap", true/* write meshes */, true/* write surfels */);
		xf::writeTransformsTextFile(framePosesWrtFrame0, outdir / "demoBkgndMap" / "frameXforms.dat");

	#if 0 //debugging
		rgbd::write_ply_file(*startFrame.getCloud(), outdir / "firstframe.ply");
		rgbd::write_ply_file(*endFrame.getCloud(), outdir / "lastframe.ply");
		cout << "bkgnd xform:" << endl << framePosesWrtFrame0.back().inverse().matrix() << endl; //TODO this should be between demo start and end frames
	#endif
	}

	/****************************************************************************************************************************************
	 * get before- and after-demo segmentations of the obj
	 */

	sumLogprobsSeparate startFrameSums(camParams.xRes * camParams.yRes);
	sumLogprobsSeparate endFrameSums(camParams.xRes * camParams.yRes);
{
	/*
	 * initial segmentation guess from diffing vs the bkgnd map
	 */
	//realign frame 0 to the current bkgnd map because the drift of visual odometry will have brought it out of alignment
	const Affine3f initialEstimateDemoStartFramePoseWrtDemoMap = initialCamPoseWrtDemoMap * framePosesWrtFrame0[demoStartFrameIndex];
	Frame frame0(*cl_ptr);
	frame0.mat_color = startFrame.getColorImg();
	frame0.mat_depth = startFrame.getDepthImg();
	demoBkgndMapper->setLastCameraPose(initialEstimateDemoStartFramePoseWrtDemoMap);
	Affine3f reoptimizedDemoStartFramePoseWrtFrame0;
	demoBkgndMapper->alignFrame(frame0, reoptimizedDemoStartFramePoseWrtFrame0);
	cout << "reopt frame 0: initialized with " << endl << initialEstimateDemoStartFramePoseWrtDemoMap.matrix() << endl;
	cout << "ended at " << endl << reoptimizedDemoStartFramePoseWrtFrame0.matrix() << endl;
	reoptimizedDemoStartFramePoseWrtFrame0 = initialCamPoseWrtDemoMap.inverse() * reoptimizedDemoStartFramePoseWrtFrame0;

	diffFrameVsPeterIntelMap(cams, startFrame.getColorImg(), startFrame.getDepthImg(), startFrame.getOrganizedCloud(), demoBkgndMapper, initialCamPoseWrtDemoMap, reoptimizedDemoStartFramePoseWrtFrame0, *noiseModelForStartEndDiffing, startFrameSums, boost::none/* extra info */);
	diffFrameVsPeterIntelMap(cams, endFrame.getColorImg(), endFrame.getDepthImg(), endFrame.getOrganizedCloud(), demoBkgndMapper, initialCamPoseWrtDemoMap, framePosesWrtFrame0[demoEndFrameIndex], *noiseModelForStartEndDiffing, endFrameSums, boost::none/* extra info */);
}
#if 1
	const cv::Mat diffingStartVisImg = visualizeLogProbsWrtFrame(camParams, startFrameSums.evidenceMoved(), startFrameSums.evidenceNotMoved(), true/* multithread */);
	cv::imwrite((outdir / "diffStartWrtBkgnd.png").string(), diffingStartVisImg);
	const cv::Mat diffingEndVisImg = visualizeLogProbsWrtFrame(camParams, endFrameSums.evidenceMoved(), endFrameSums.evidenceNotMoved(), true/* multithread */);
	cv::imwrite((outdir / "diffEndWrtBkgnd.png").string(), diffingEndVisImg);
#endif

	/****************************************************************************************************************************************
	 * refine segmentations, and get a rigid-xform estimate, with ransac on high-p(m) points in the first and last frames of the demo
	 */

	const rgbd::eigen::VectorXd startFrameMovedProbs = startFrameSums.movedProbs(), endFrameMovedProbs = endFrameSums.movedProbs();
	const pcl::PointCloud<rgbd::pt> startCloud = *startFrame.getCloud(), endCloud = *endFrame.getCloud();
	const auto chooseAllSeeds = [](const rgbdFrame& frame, const rgbd::eigen::VectorXd& frameMovedProbs)
		{
			const rgbd::CameraParams camParams = frame.camParams;
			const pcl::PointCloud<rgbd::pt> frameCloud = *frame.getCloud();
			std::vector<size_t> allSeedPts;
#if 1 //don't allow pts near depth boundaries as seeds
			const cv::Mat_<float>& depth = frame.getDepthImg();
			cv::Mat_<uint8_t> invalidMask(depth.rows, depth.cols); //mark pixels with invalid depth or at depth boundaries with 0
			for(size_t i = 0; i < camParams.yRes; i++)
				for(size_t j = 0; j < camParams.xRes; j++)
				{
					if(depth(i, j) <= 0) invalidMask(i, j) = 0;
					else if((int32_t)i < depth.rows - 1 && depth(i + 1, j) > 0 && fabs(depth(i, j) - depth(i + 1, j)) > .02/* TODO ? */ * primesensor::stereoErrorRatio(std::min(depth(i, j), depth(i + 1, j)))) invalidMask(i, j) = 0;
					else if((int32_t)j < depth.cols - 1 && depth(i, j + 1) > 0 && fabs(depth(i, j) - depth(i, j + 1)) > .02/* TODO ? */ * primesensor::stereoErrorRatio(std::min(depth(i, j), depth(i, j + 1)))) invalidMask(i, j) = 0;
					else invalidMask(i, j) = 255;
				}
			cv::Mat distsToInvalid;
			cv::distanceTransform(invalidMask, distsToInvalid, CV_DIST_L1, CV_DIST_MASK_PRECISE);
#endif
			for(size_t i = 0; i < frameCloud.points.size(); i++)
			{
				const rgbd::pt pt = frameCloud.points[i];
				if(frameMovedProbs[pt.imgY * camParams.xRes + pt.imgX] > .52/* TODO ? */
					&& distsToInvalid.at<float>(pt.imgY, pt.imgX) > 1.9/* TODO ? */
					)
					allSeedPts.push_back(i);
			}
			return allSeedPts;
		};
	const std::vector<size_t> allSeedPtsBefore = chooseAllSeeds(startFrame, startFrameMovedProbs), allSeedPtsAfter = chooseAllSeeds(endFrame, endFrameMovedProbs); //indices into unorganized clouds
	/*
	 * select a subset of seed pts evenly distributed in space and not too close together
	 */
	const float minSeedDist = .03; //min dist btwn allowable seeds, in m; TODO tweakable
	std::vector<size_t> seedPtsBefore;
	const auto pickAllowableSeeds = [minSeedDist](rgbdFrame& frame, const std::vector<size_t>& allSeedPts, std::vector<size_t>& seedPts)
		{
			pcl::PointCloud<rgbd::pt> frameSubset;
			rgbd::downsampleFromIndices(*frame.getCloud(), frameSubset, allSeedPts);
			const boost::shared_ptr<kdtree2> kdtree = rgbd::createKDTree2(frameSubset);
			const kdtreeNbrhoodSpec nspec = kdtreeNbrhoodSpec::byRadius(minSeedDist);
			std::unordered_set<size_t> allSeedPtsLeft(allSeedPts.begin(), allSeedPts.end());
			while(!allSeedPtsLeft.empty())
			{
				const size_t j = *allSeedPtsLeft.begin();
				seedPts.push_back(j);
				const rgbd::pt pt = (*frame.getCloud()).points[j];
				const std::vector<float> qpt = {pt.x, pt.y, pt.z};
				const std::vector<kdtree2_result> nbrs = searchKDTree(*kdtree, nspec, qpt);
				for(const auto n : nbrs) allSeedPtsLeft.erase(allSeedPts[n.idx]);
			}
		};
	pickAllowableSeeds(startFrame, allSeedPtsBefore, seedPtsBefore);
	std::unordered_set<size_t> allSeedPtsAfterSet(allSeedPtsAfter.begin(), allSeedPtsAfter.end());
#if 0
	pcl::PointCloud<rgbd::pt> seedsCloud = startCloud;
	for(auto& pt : seedsCloud.points) pt.rgb = rgbd::packRGB(0, 0, 155);
	for(size_t i : seedPtsBefore) seedsCloud.points[i].rgb = rgbd::packRGB(255, 0, 0);
	rgbd::write_ply_file(seedsCloud, outdir / "seedsBefore.ply");
#endif

	/*
	 * reoptimize various alignments as necessary
	 */
	Affine3f reoptimizedDemoEndFramePoseWrtDemoStartFrame;
{
#if 1 //this helps a lot 20130904
	//refine the xform from peter-intel mapping, since it might be pretty bad due to our abuse of the static mapping system to align with a non-static map
	std::shared_ptr<VolumeModeler> frame0mapper(new VolumeModeler(clKernels, modelerParams));
	Frame frame0(*cl_ptr);
	frame0.mat_color = startFrame.getColorImg();
	frame0.mat_depth = startFrame.getDepthImg();
	frame0.mat_segments = cv::Mat(frame0.mat_color.size(), CV_32S, cv::Scalar(0));
	Affine3f demoStartCamPoseWrtTmpMap;
	frame0mapper->alignFrame(frame0, demoStartCamPoseWrtTmpMap);
	frame0mapper->addFrame(frame0, demoStartCamPoseWrtTmpMap);
	frame0mapper->setLastCameraPose(demoStartCamPoseWrtTmpMap * framePosesWrtFrame0[demoStartFrameIndex].inverse() * framePosesWrtFrame0[demoEndFrameIndex]);
	Frame frameEnd(*cl_ptr);
	frameEnd.mat_color = endFrame.getColorImg();
	frameEnd.mat_depth = endFrame.getDepthImg();
	frameEnd.mat_segments = cv::Mat(frameEnd.mat_color.size(), CV_32S, cv::Scalar(0));
	Affine3f demoEndCamPoseWrtTmpMap;
	frame0mapper->alignFrame(frameEnd, demoEndCamPoseWrtTmpMap);
	frame0mapper.reset(); //free memory
	reoptimizedDemoEndFramePoseWrtDemoStartFrame = demoStartCamPoseWrtTmpMap.inverse() * demoEndCamPoseWrtTmpMap;
	cout << "new start2end xform:" << endl << reoptimizedDemoEndFramePoseWrtDemoStartFrame.inverse().matrix() << endl;
#else
	reoptimizedDemoEndFramePoseWrtDemoStartFrame = framePosesWrtFrame0.back(); TODO this is wrong if demo doesn't start at frame 0
#endif
}

	/*
	 * set up for ransac: get correspondences between pts in demo start and end frames
	 */
	const float maxObjMoveDist = .05; //max distance any seed pt can have moved during the demo; TODO parameterize
	std::vector<rgbd::eigen::Vector3f> srcPts, tgtPts; //all pts of the unorganized clouds
	convertPointCloudToVector(startCloud, srcPts);
	convertPointCloudToVector(endCloud, tgtPts);
	//allow matching only to tgt pts that are close in 3-d after applying the start-to-end rigid (bkgnd) xform
	std::vector<std::pair<int, int> > allCorrs;

	/*
	 * render the end frame into the start frame
	 */
	boost::multi_array<uint32_t, 2> ptIDs(boost::extents[camParams.yRes][camParams.xRes]);
	boost::multi_array<float, 2> ptDepths(boost::extents[camParams.yRes][camParams.xRes]);
	projectPointsIntoCamera(tgtPts, camParams, reoptimizedDemoEndFramePoseWrtDemoStartFrame.inverse(), ptIDs, ptDepths);
#define RENDER_PT_CLOUD //or render the frame from a peter-intel map of just that frame
#ifdef RENDER_PT_CLOUD
	cv::Mat_<cv::Vec3b> bgrImg(camParams.yRes, camParams.xRes);
	cv::Mat_<float> renderedEndFrameDepth(camParams.yRes, camParams.xRes);
	for(size_t i = 0, l = 0; i < camParams.yRes; i++)
		for(size_t j = 0; j < camParams.xRes; j++, l++)
		{
			if(ptIDs[i][j] > 0)
			{
				const boost::array<uint8_t, 3> rgb = rgbd::unpackRGB<uint8_t>(endCloud.points[ptIDs[i][j] - 1].rgb);
				for(int k = 0; k < 3; k++) bgrImg(i, j)[k] = clamp((int)rgb[2 - k] - 4 + rand() % 9, 0, 255); /* add slight color noise to avoid zero gradients for flow */
				renderedEndFrameDepth(i, j) = ptDepths[i][j];
			}
			else
			{
				renderedEndFrameDepth(i, j) = 0;
			}
		}
#else
	std::shared_ptr<VolumeModeler> frame0mapper(new VolumeModeler(*clKernels.get(), params_volume_modeler, params_camera, params_volume, params_features, params_optimize, params_normals, params_grid, params_loop_closure));
	Frame frameEnd(*cl_ptr);
	frameEnd.mat_color = endFrame.getColorImg();
	frameEnd.mat_depth = endFrame.getDepthImg();
	frameEnd.mat_segments = cv::Mat(frameEnd.mat_color.size(), CV_32S, cv::Scalar(0));
	Affine3f tmpEndXform;
	frame0mapper->alignFrame(frameEnd, tmpEndXform);
	frame0mapper->addFrame(frameEnd, tmpEndXform);
	frame0mapper->render(tmpEndXform * reoptimizedEnd2startXform.inverse());
	cv::Mat_<cv::Vec4b> renderedCols(camParams.yRes, camParams.xRes, cv::DataType<cv::Vec4b>::type); //bgra
	std::vector<float> renderedPts(camParams.yRes * camParams.xRes * 4);
	std::vector<float> renderedNormals(camParams.yRes * camParams.xRes * 4);
	std::vector<int> renderingValidity(camParams.yRes * camParams.xRes); //iff 0 at a pixel, pts, normals and maybe colors are nan
	frame0mapper->getLastRender(renderedCols, renderedPts, renderedNormals, renderingValidity);
	cv::Mat_<cv::Vec3b> bgrImg(renderedCols.rows, renderedCols.cols);
	const int fromTo[6] = {0, 0,  1, 1,  2, 2};
	cv::mixChannels(&renderedCols, 1, &bgrImg, 1, fromTo, 3);
	cv::Mat_<float> renderedEndFrameDepth(renderedCols.rows, renderedCols.cols);
	for(size_t i = 0, l = 0; i < camParams.yRes; i++)
		for(size_t j = 0; j < camParams.xRes; j++, l++)
			if(!renderingValidity[l]) renderedEndFrameDepth(i, j) = 0;
			else renderedEndFrameDepth(i, j) = renderedPts[l * 4 + 2];
	frame0mapper.reset(); //free memory
#endif
#if 1
	cv::imwrite((outdir / "renderedBeforeHoleFill.png").string(), bgrImg);
#endif
	/*
	 * fill in holes to give flow a nicer img to work with
	 */
	std::unordered_set<size_t> unsetPixels;
	for(size_t i = 0, l = 0; i < camParams.yRes; i++)
		for(size_t j = 0; j < camParams.xRes; j++, l++)
			if(renderedEndFrameDepth(i, j) == 0)
				unsetPixels.insert(l);
	while(!unsetPixels.empty())
	{
		std::unordered_set<size_t> toRemove;
		for(size_t l : unsetPixels)
		{
			bool set = false;
#define CHECK(yy, xx) \
			if(unsetPixels.find(yy * camParams.xRes + xx) == unsetPixels.end())\
			{\
				/* copy, & add slight color noise to avoid zero gradients for flow */\
				for(int p = 0; p < 3; p++) bgrImg(l / camParams.xRes, l % camParams.xRes)[p] = clamp((int)bgrImg(yy, xx)[p] - 4 + rand() % 9, 0, 255);\
				renderedEndFrameDepth(l / camParams.xRes, l % camParams.xRes) = renderedEndFrameDepth(yy, xx);\
				toRemove.insert(l);\
			}

			const int32_t y = l / camParams.xRes, x = l % camParams.xRes;
			if(y > 0) CHECK((y - 1), x);
			if(!set && x > 0) CHECK(y, (x - 1));
			if(!set && y < (int32_t)camParams.yRes - 1) CHECK((y + 1), x);
			if(!set && x < (int32_t)camParams.xRes - 1) CHECK(y, (x + 1));
#undef CHECK
		}
		for(size_t l : toRemove) unsetPixels.erase(l);
	}
#if 1
	cv::imwrite((outdir / "renderedAfterHoleFill.png").string(), bgrImg);
#endif

	cv::Mat_<cv::Vec3b> img0small = startFrame.getColorImg().clone(), img1small = bgrImg;
	cv::Mat_<float> depth0small = startFrame.getDepthImg().clone(), depth1small = renderedEndFrameDepth;
	downsizeFrame(img0small, depth0small);
	downsizeFrame(img1small, depth1small);
	const cv::Mat_<cv::Vec3f> sceneFlowSmall = runFlowTwoFramesSceneFlowExperiments(camParams, img0small, depth0small, img1small, depth1small, .02/* smoothnessWeight */, 1/* depthVsColorDataWeight */, OpticalFlow::sceneFlowRegularizationType::HPB, 1000/* max iters */, boost::none/* estimate */, boost::none/* outdirPlusFilebase */);
	cv::Mat_<cv::Vec3f> sceneFlow(camParams.yRes, camParams.xRes);
	cv::resize(sceneFlowSmall, sceneFlow, sceneFlow.size());
#if 1
	writeSceneFlow(sceneFlow, outdir / "flow.flo3");
#endif

	//pick the nearest tgt pt for each (src pt + flow)
	const boost::shared_ptr<kdtree2> endTree = rgbd::createKDTree2(endCloud);
	for(size_t i = 0; i < seedPtsBefore.size(); i++)
	{
		const float u1 = startCloud.points[seedPtsBefore[i]].imgX, v1 = startCloud.points[seedPtsBefore[i]].imgY;
		const float u2 = u1 + sceneFlow(v1, u1)[0], v2 = v1 + sceneFlow(v1, u1)[1];
		if(ptIDs[(int)rint(v2)][(int)rint(u2)] > 0 && fabs((startCloud.points[seedPtsBefore[i]].z + sceneFlow(v1, u1)[2]) - ptDepths[(int)rint(v2)][(int)rint(u2)]) < .01/* TODO ? */)
			allCorrs.push_back(std::make_pair<int, int>(seedPtsBefore[i], ptIDs[(int)rint(v2)][(int)rint(u2)] - 1));
	}

	const float minSampleDist = .04, maxSampleDist = .2; //TODO ?
	registration::ransacDenseUniqueCorrespondenceSampler seedSampler(srcPts, allCorrs, minSampleDist, maxSampleDist);
	const auto sampleFunc = [&](std::vector<unsigned int>& sampleIndices) -> bool
		{
			return seedSampler.sample(1000/* max attempts */, srcPts, allCorrs, sampleIndices);
		};

	/*
	 * set up diffing for scoring candidate transforms
	 */
	std::shared_ptr<openglContext> glContext(new openglContext(camParams.xRes, camParams.yRes));
	std::shared_ptr<viewScoringRenderer> sceneRenderer(new viewScoringRenderer(glContext, camParams, false/* use large texture */));
	std::shared_ptr<sceneSampler> sampler1(new sceneSamplerPointCloud(startCloud)), sampler2(new sceneSamplerPointCloud(endCloud));
	std::shared_ptr<pointCloudRenderer> samplesCache1(new pointCloudRenderer(startCloud, getPointColorFromID, 1/* pt size */, camParams)),
			samplesCache2(new pointCloudRenderer(endCloud, getPointColorFromID, 1/* pt size */, camParams));
	diffingSingleFrameInfo startFrameInfo, endFrameInfo;
	startFrameInfo.sceneName = "scene1";
	startFrameInfo.frameDepth = startFrame.getDepthImg();
	computeDepthMapLocalStdevs(startFrame.getDepthImg(), 4/* nbrhoodHalfwidth; TODO ? */, startFrameInfo.depthStdevs, 4/* # pixels near depth edges to increase uncertainty for; TODO ? */, camParams, true/* multithread */);
	startFrameInfo.frameImg = startFrame.getColorImg();
	startFrameInfo.frameNormalsCloud = startFrame.getOrganizedCloud(); //TODO use a multiarray for diffing speed
	startFrameInfo.framePoseWrtMap = Affine3f::Identity();
	endFrameInfo.sceneName = "scene2";
	endFrameInfo.frameDepth = endFrame.getDepthImg();
	computeDepthMapLocalStdevs(endFrame.getDepthImg(), 4/* nbrhoodHalfwidth; TODO ? */, endFrameInfo.depthStdevs, 4/* # pixels near depth edges to increase uncertainty for; TODO ? */, camParams, true/* multithread */);
	endFrameInfo.frameImg = endFrame.getColorImg();
	endFrameInfo.frameNormalsCloud = endFrame.getOrganizedCloud(); //TODO use a multiarray for diffing speed
	endFrameInfo.framePoseWrtMap = Affine3f::Identity();
	const projectSamplesIntoCameraPerPixelFunc sampleProjectionFunc = [&sceneRenderer](const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose, boost::multi_array<uint32_t, 2>& sampleIDs, boost::multi_array<float, 2>& sampleDepths, cv::Mat_<float>& sampleDepthStdevs, bool& stdevsSet)
		{
			projectSceneSamplesIntoCamera(*sceneRenderer, camParams, camPose, sampleIDs, sampleDepths);
		};
	sceneDifferencer differ1(cams), differ2(cams);
	differ1.setPrevScene(sampler1);
	differ2.setPrevScene(sampler2);
	singleFrameDifferencingParams diffingParams;
	const double posWeight = 1, colWeight = .5, normalWeight = 0; //TODO ?

	/************************************************************************
	 * ransac main loop
	 */
	float bestScore = -FLT_MAX, bestBadScoreBefore = -FLT_MAX, bestBadScoreAfter = -FLT_MAX, bestGoodScoreBefore = -FLT_MAX, bestGoodScoreAfter = -FLT_MAX; //basically count likely-obj pts that like or don't like this xform
	Affine3f bestXform;
	rgbd::eigen::VectorXd bestMovedProbsBefore, bestMovedProbsAfter;
	uint64_t iter = 0;
	uint64_t itersSinceImprovement = 0;
	while(std::min(bestGoodScoreBefore, bestGoodScoreAfter) < .6/* TODO ? */ || std::min(bestBadScoreBefore, bestBadScoreAfter) < -.035/* TODO ? */ || itersSinceImprovement < 20/* TODO ? */)
	{
		rgbd::timer t;
		/*
		 * propose an xform
		 */
		std::vector<unsigned int> sampleIndices(3/* sample size */);
		const bool success = sampleFunc(sampleIndices);
		if(success)
		{
			std::deque<std::pair<rgbd::eigen::Vector3f,rgbd::eigen::Vector3f> > corrPts(sampleIndices.size());
			for(size_t i = 0; i < sampleIndices.size(); i++)
			{
				corrPts[i].first = srcPts[allCorrs[sampleIndices[i]].first];
				corrPts[i].second = tgtPts[allCorrs[sampleIndices[i]].second];
			}
			rgbd::eigen::Quaternionf rot;
			rgbd::eigen::Vector3f trans;
			const std::vector<float> corrWeights(corrPts.size(), 1);
			registration::runClosedFormAlignment(corrPts, corrWeights, rot, trans);
			const Affine3f xform = rgbd::eigen::Translation3f(trans) * rot; //start frame -> end frame

			/*
			 * score with diffing
			 */

			sumLogprobsSeparate sumsEndFrame(camParams.yRes * camParams.xRes);
		{
			noiseModelForMofitting->cachePerPointInfo(*sampler1); //TODO use two noise models for the two samplers, for speed?
			sceneRenderer->acquire();
			sceneRenderer->setRenderFunc([&samplesCache1](const rgbd::eigen::Affine3f& camPose){samplesCache1->render(camPose);});
			const Affine3f framePoseWrtMap = xform.inverse();
			differ1.setCurFrame(endFrameInfo, framePoseWrtMap); //TODO can set these separately for speed?
			differ1.projectSceneIntoFrame(sampleProjectionFunc);
			diffingParams.differenceFrameWrtCloud = false;
			diffingParams.outputWrtCloud = false;
			differ1.runDifferencingAfterProjection(*noiseModelForMofitting, sumsEndFrame, diffingParams);
			sumsEndFrame.weightComponents(posWeight, colWeight, normalWeight);
			sceneRenderer->restoreRenderFunc();
			sceneRenderer->release();
		}

			sumLogprobsSeparate sumsStartFrame(camParams.yRes * camParams.xRes);
		{
			noiseModelForMofitting->cachePerPointInfo(*sampler2);
			sceneRenderer->acquire();
			sceneRenderer->setRenderFunc([&samplesCache2](const rgbd::eigen::Affine3f& camPose){samplesCache2->render(camPose);});
			const Affine3f framePoseWrtMap = xform;
			differ2.setCurFrame(startFrameInfo, framePoseWrtMap); //TODO can set these separately for speed?
			differ2.projectSceneIntoFrame(sampleProjectionFunc);
			diffingParams.differenceFrameWrtCloud = false;
			diffingParams.outputWrtCloud = false;
			differ2.runDifferencingAfterProjection(*noiseModelForMofitting, sumsStartFrame, diffingParams);
			sumsStartFrame.weightComponents(posWeight, colWeight, normalWeight);
			sceneRenderer->restoreRenderFunc();
			sceneRenderer->release();
		}

			rgbd::timer t3;
			cv::Mat_<cv::Vec3b> diffImg = visualizeLogProbsWrtFrame(camParams, sumsStartFrame.evidenceMoved(), sumsStartFrame.evidenceNotMoved(), true/* multithread */);
			for(size_t i = 0; i < 3; i++) cv::circle(diffImg, cv::Point(startCloud.points[allCorrs[sampleIndices[i]].first].imgX, startCloud.points[allCorrs[sampleIndices[i]].first].imgY), 2, cv::Scalar(0, 255, 0), -1);
			cv::imwrite((outdir / (boost::format("diffRANSACBefore%1%.png") % iter).str()).string(), diffImg);
			diffImg = visualizeLogProbsWrtFrame(camParams, sumsEndFrame.evidenceMoved(), sumsEndFrame.evidenceNotMoved(), true/* multithread */);
			for(size_t i = 0; i < 3; i++) cv::circle(diffImg, cv::Point(endCloud.points[allCorrs[sampleIndices[i]].second].imgX, endCloud.points[allCorrs[sampleIndices[i]].second].imgY), 2, cv::Scalar(0, 255, 0), -1);
			cv::imwrite((outdir / (boost::format("diffRANSACAfter%1%.png") % iter).str()).string(), diffImg);
			t3.stop("write imgs");

			/*
			 * compute a score using diffing results
			 */
			rgbd::timer t2;
			const rgbd::eigen::VectorXd movedProbsStartFrame = sumsStartFrame.movedProbs(), movedProbsEndFrame = sumsEndFrame.movedProbs();
			const rgbd::eigen::VectorXd evidenceMagnitudeStartFrame = sumsStartFrame.evidenceMagnitude(), evidenceMagnitudeEndFrame = sumsEndFrame.evidenceMagnitude();
			float goodScoreBefore = 0, badScoreBefore = 0, goodScoreAfter = 0, badScoreAfter = 0;
			for(size_t i : allSeedPtsBefore) //for pixels that moved wrt the bkgnd
			{
				const size_t l = startCloud.points[i].imgY * camParams.xRes + startCloud.points[i].imgX; //TODO cache these
				badScoreBefore += std::min(0.0, .5 - movedProbsStartFrame[l]);
				if(movedProbsStartFrame[l] < .5) goodScoreBefore += 1;//std::max(0.0, .5 - movedProbsBefore[l]);
			}
			for(size_t i : allSeedPtsAfter) //for pixels that moved wrt the bkgnd
			{
				const size_t l = endCloud.points[i].imgY * camParams.xRes + endCloud.points[i].imgX; //TODO cache these
				badScoreAfter += std::min(0.0, .5 - movedProbsEndFrame[l]);
				if(movedProbsEndFrame[l] < .5) goodScoreAfter += 1;//std::max(0.0, .5 - movedProbsAfter[l]);
			}
			const float score = (goodScoreBefore + goodScoreAfter + badScoreBefore + badScoreAfter) / (allSeedPtsBefore.size() + allSeedPtsAfter.size());
			badScoreBefore /= allSeedPtsBefore.size();
			badScoreAfter /= allSeedPtsAfter.size();
			goodScoreBefore /= allSeedPtsBefore.size();
			goodScoreAfter /= allSeedPtsAfter.size();
			cout << "scores " << iter << ": " << goodScoreBefore << ' ' << badScoreBefore << ' ' << goodScoreAfter << ' ' << badScoreAfter << endl;
			cout << " best " << iter << ": " << bestGoodScoreBefore << ' ' << bestBadScoreBefore << ' ' << bestGoodScoreAfter << ' ' << bestBadScoreAfter << endl;
			t2.stop("compute scores");
			if(score > bestScore)
			{
				bestScore = score;
				bestGoodScoreBefore = goodScoreBefore;
				bestBadScoreBefore = badScoreBefore;
				bestGoodScoreAfter = goodScoreAfter;
				bestBadScoreAfter = badScoreAfter;
				bestXform = xform;
				bestMovedProbsBefore = movedProbsStartFrame;
				bestMovedProbsAfter = movedProbsEndFrame;
				itersSinceImprovement = 0;
			}
		}
		t.stop("run one ransac iter");
		iter++;
		itersSinceImprovement++;
	}
	cout << "bkgnd xform:" << endl << reoptimizedDemoEndFramePoseWrtDemoStartFrame.inverse().matrix() << endl;
	cout << "obj xform:" << endl << bestXform.matrix() << endl;

	/****************************************************************************************************************************************
	 * spatially regularize to get a final object segmentation in the before- and after-demo frames, using all the information we have
	 */

	const auto spatiallyRegularizeFinalObjSegmentation = [](const rgbd::CameraParams& camParams, const cv::Mat_<float>& depthImg, const rgbd::eigen::VectorXd& frameMovedProbsWrtBkgndMap, const rgbd::eigen::VectorXd& frameMovedProbsWrtEstimatedObjMotion,
		const float probMovedThreshold)
		{
			rgbd::timer t;
			const float smoothnessWeight = 1e1; //TODO ?
			const size_t numLabels = 2; //states: 0 = bg, 1 = obj

			typedef GCoptimizationGeneralGraph::SiteID SiteID;
			typedef GCoptimizationGeneralGraph::LabelID LabelID;
			typedef GCoptimizationGeneralGraph::EnergyTermType EnergyTermType;
			typedef GCoptimizationGeneralGraph::EnergyType EnergyType;

			class segmDemoEndObjsegDataCostFunctor : public GCoptimization::DataCostFunctor
			{
				public:

					segmDemoEndObjsegDataCostFunctor(const rgbd::eigen::VectorXd& mwbg, const rgbd::eigen::VectorXd& mwom, const float pmt)
					: movedProbsWrtBkgndMap(mwbg), movedProbsWrtEstimatedObjMotion(mwom), probMovedThreshold(pmt)
					{}
					virtual ~segmDemoEndObjsegDataCostFunctor() {}

					virtual EnergyTermType compute(SiteID s, LabelID l)
					{
						EnergyTermType energy = 0;
						const double probMovedWrtBkgnd = std::min(1 - 1e-7, std::max(1e-7, movedProbsWrtBkgndMap[s])), probMovedWrtMotion = std::min(1 - 1e-7, std::max(1e-7, movedProbsWrtEstimatedObjMotion[s]));
						switch(l)
						{
							case 0: //bg
								energy = -log(1 - probMovedWrtBkgnd);
								break;
							case 1: //obj: want it to have moved wrt bkgnd and not moved wrt estimated obj motion
								energy = -log(probMovedWrtBkgnd) + -log(1 - (probMovedWrtMotion - .495/* give evidenceless pixels a slight bias toward bg */));
								break;
						}
						if(energy > GCO_MAX_ENERGYTERM) energy = GCO_MAX_ENERGYTERM; //avoid GCO errors
						return energy;
					}

				private:

					const rgbd::eigen::VectorXd& movedProbsWrtBkgndMap;
					const rgbd::eigen::VectorXd& movedProbsWrtEstimatedObjMotion;
					const float probMovedThreshold;
			};
			class segmDemoEndObjsegSmoothnessCostFunctor : public GCoptimization::SmoothCostFunctor
			{
				public:

					segmDemoEndObjsegSmoothnessCostFunctor(const double w) : weight(w)
					{}
					virtual ~segmDemoEndObjsegSmoothnessCostFunctor() {}

					virtual EnergyTermType compute(SiteID s1, SiteID s2, LabelID l1, LabelID l2)
					{
						/*
						 * the interesting part of the smoothness cost is in the edge weights
						 */
						return (l1 != l2) ? weight : 0;
					}

				private:

					const double weight;
			};

			GCoptimizationGridGraph graph(camParams.xRes, camParams.yRes, numLabels);
			vector<SiteID> numNbrs(camParams.yRes * camParams.xRes);
			vector<SiteID*> nbrLists(camParams.yRes * camParams.xRes);
			boost::multi_array<EnergyTermType, 2> smoothnessCostByLabels(boost::extents[numLabels][numLabels]); //TODO this bit doesn't seem to be used, because I need to call setSmoothCostFunctor() later for gc to work -- ??
			smoothnessCostByLabels[0][0] = 0;
			smoothnessCostByLabels[0][1] = smoothnessWeight;
			smoothnessCostByLabels[1][0] = 0;
			smoothnessCostByLabels[1][1] = smoothnessWeight;
			vector<EnergyTermType> vertNbrLinkWeights(camParams.yRes * camParams.xRes), horizNbrLinkWeights(camParams.yRes * camParams.xRes); //from each pixel to its right and down nbrs
			for(size_t i = 0, l = 0; i < camParams.yRes; i++)
				for(size_t j = 0; j < camParams.xRes; j++, l++)
				{
		#define SET_WEIGHT(i, j, i2, j2, wt) \
			wt = exp(-sqr(fabs(depthImg(i, j) - depthImg(i2, j2))) / sqr(.005/* TODO ? */ * std::max(1.0, std::min(primesensor::stereoErrorRatio(depthImg(i, j)), primesensor::stereoErrorRatio(depthImg(i2, j2)))))); //TODO can I do something smarter w/ invalid depths?
					if(i < camParams.yRes - 1) SET_WEIGHT(i, j, (i + 1), j, vertNbrLinkWeights[l]);
					if(j < camParams.xRes - 1) SET_WEIGHT(i, j, i, (j + 1), horizNbrLinkWeights[l]);
		#undef SET_WEIGHT
				}
			graph.setSmoothCostVH(smoothnessCostByLabels.data(), vertNbrLinkWeights.data(), horizNbrLinkWeights.data());
			graph.setDataCostFunctor(new segmDemoEndObjsegDataCostFunctor(frameMovedProbsWrtBkgndMap, frameMovedProbsWrtEstimatedObjMotion, probMovedThreshold));
			graph.setSmoothCostFunctor(new segmDemoEndObjsegSmoothnessCostFunctor(smoothnessWeight));
			graph.setLabelCost(0.0f);
			t.stop("set up graph cuts");

			const uint32_t maxIters = 10;
			t.restart();
			graph.expansion(maxIters);
			t.stop("run graph cuts");

			cv::Mat_<uint8_t> objMaskInFrame(camParams.yRes, camParams.xRes);
			for(size_t i = 0, l = 0; i < camParams.yRes; i++)
				for(size_t j = 0; j < camParams.xRes; j++, l++)
					switch(graph.whatLabel(l))
					{
						case 0: //bg
							objMaskInFrame(i, j) = 0;
							break;
						case 1: //obj
							objMaskInFrame(i, j) = 255;
							break;
						default: ASSERT_ALWAYS(false);
					}
			return objMaskInFrame;
		};

	cv::Mat_<uint8_t> objMaskInDemoStartFrame = spatiallyRegularizeFinalObjSegmentation(camParams, startFrame.getDepthImg(), startFrameMovedProbs, bestMovedProbsBefore, probMovedThreshold),
		objMaskInDemoEndFrame = spatiallyRegularizeFinalObjSegmentation(camParams, endFrame.getDepthImg(), endFrameMovedProbs, bestMovedProbsAfter, probMovedThreshold);
#if 1 //visualize
	cv::imwrite((outdir / "objMaskMRFStart.png").string(), objMaskInDemoStartFrame);
	cv::imwrite((outdir / "objMaskMRFEnd.png").string(), objMaskInDemoEndFrame);
#endif

	/****************************************************************************************************************************************
	 * adjust obj masks by removing small 3-d conn comps to get final before- and after-demo obj segs
	 */

	rgbd::eigen::VectorXd objMaskStartVec(camParams.yRes * camParams.xRes), objMaskEndVec(camParams.yRes * camParams.xRes);
	for(size_t i = 0, l = 0; i < camParams.yRes; i++)
		for(size_t j = 0; j < camParams.xRes; j++, l++)
		{
			switch(objMaskInDemoStartFrame(i, j))
			{
				case 0: //bg
					objMaskStartVec[l] = 0;
					break;
				case 255: //obj
					objMaskStartVec[l] = 1;
					break;
				default: ASSERT_ALWAYS(false);
			}
			switch(objMaskInDemoEndFrame(i, j))
			{
				case 0: //bg
					objMaskEndVec[l] = 0;
					break;
				case 255: //obj
					objMaskEndVec[l] = 1;
					break;
				default: ASSERT_ALWAYS(false);
			}
		}
	//these calls will return only the conn comps with high objMaskVec value; make sure that corresponds to being a hypothetical obj comp rather than a bkgnd comp
	const float maxIntersegDist = .1; //in 3-d, in m; TODO ?
	std::unordered_map<size_t, std::vector<size_t>> connCompsStart = std::move(findDiffingConnComps(camParams, startFrame.getDepthImg(), objMaskStartVec, .5/*probMovedThreshold*/, diffingConnCompsAlgorithm::FULL_3D, maxIntersegDist)),
		connCompsEnd = std::move(findDiffingConnComps(camParams, endFrame.getDepthImg(), objMaskEndVec, .5/*probMovedThreshold*/, diffingConnCompsAlgorithm::FULL_3D, maxIntersegDist));

	//remove small comps
	const size_t minCompSizeToKeep = 4000; //TODO ?
{
	std::vector<size_t> compsToRemove;
	for(const auto& c : connCompsStart)
		if(c.second.size() < minCompSizeToKeep)
			compsToRemove.push_back(c.first);
	for(size_t c : compsToRemove) connCompsStart.erase(c);
}
{
	std::vector<size_t> compsToRemove;
	for(const auto& c : connCompsEnd)
		if(c.second.size() < minCompSizeToKeep)
			compsToRemove.push_back(c.first);
	for(size_t c : compsToRemove) connCompsEnd.erase(c);
}

#if 1 //visualize start- & end-demo obj segs
	cv::Mat_<cv::Vec3b> startCompsImg(camParams.yRes, camParams.xRes), endCompsImg(camParams.yRes, camParams.xRes);
	const std::vector<boost::array<uint8_t, 3>> compCols = getDistinguishableColors(std::max(connCompsStart.size(), connCompsEnd.size()) + 1);
	const boost::array<uint8_t, 3> bkgndCol = compCols.back();
	for(size_t i = 0, l = 0; i < camParams.yRes; i++)
		for(size_t j = 0; j < camParams.xRes; j++, l++)
		{
			startCompsImg(i, j) = cv::Vec3b(bkgndCol[2], bkgndCol[1], bkgndCol[0]);
			endCompsImg(i, j) = cv::Vec3b(bkgndCol[2], bkgndCol[1], bkgndCol[0]);
		}
	size_t i0 = 0;
	for(auto i = connCompsStart.begin(); i != connCompsStart.end(); i++, i0++)
		for(const size_t p : (*i).second)
			startCompsImg(p / camParams.xRes, p % camParams.xRes) = cv::Vec3b(compCols[i0][2], compCols[i0][1], compCols[i0][0]);
	i0 = 0;
	for(auto i = connCompsEnd.begin(); i != connCompsEnd.end(); i++, i0++)
		for(const size_t p : (*i).second)
			endCompsImg(p / camParams.xRes, p % camParams.xRes) = cv::Vec3b(compCols[i0][2], compCols[i0][1], compCols[i0][0]);
	cv::imwrite((outdir / "objMaskCompsStart.png").string(), startCompsImg);
	cv::imwrite((outdir / "objMaskCompsEnd.png").string(), endCompsImg);
#endif

	//TODO start adding to a new obj modeler w/ the end-of-demo frame (or perhaps also add the before-demo frame)
	// (or can create an initial obj modeler from diffing the before-demo map in 3-d wrt the final demo bkgnd map, by comparing tsdf values directly; probably can't rely on the same techniques we use for empty-bkgnd-map modeling)
	//TODO merge demo bkgnd map into general bkgnd map using mergeOtherVolume()
	//TODO merge the new obj model with any old one(s) it overlaps with (how?)

	return 0;
}
