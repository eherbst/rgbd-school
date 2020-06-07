/*
 * onlineActiveModeler: active but non-interactive object modeling for an icra14 submission
 *
 * Evan Herbst
 * 7 / 13 / 13
 */

//#define USE_SPHERE_CARVING //to carve arbitrary polyhedra into a peter-intel map; not needed once peter's code supports bbox carving

#define USE_CGAL_COLLISION_CHECKING //as opposed to openrave

#define USE_APPROX_ROBOT_MESHES_FOR_COLLISION_CHECKING //as opposed to full meshes, use a reduced and/or approximate triangle set; orthogonal to the cgal/non-cgal collision checking choice

/*
 * else use openrave's default planner
 *
 * 20131104 the openrave birrt finds a legal path more often than chomp does; possibly the problem is that chomp is harder to use
 */
#define PLAN_WITH_CHOMP

#ifdef PLAN_WITH_CHOMP
/*
 * if not defined, use chomp's default method of making an sdf; if defined, use ours on the gpu, which is ~8x as fast
 *
 * if using grid maps we must use chomp's method as of 20131205; to fix this we could copy the grid into a single volume of just the size we need and copy its sdf for chomp
 */
#define MAKE_MAP_MESH_SDF_OURSELVES
#endif

#define USE_OMPL //or use our own rrt (for free-flying cameras, where openrave isn't usable)

//#define DEBUG_POSE_SUGGESTION //uncomment to visualize various imgs/clouds of poses
//#define DEBUG_PLANNING //uncomment to visualize planned paths

//for 20140415 experiment seeing whether diffing-based view selection does better on objs than regular selection; TODO remove altogether
//#define RUN_DIFFING_SCORING_EXPERIMENT

#include <signal.h> //raise()
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/functional/hash.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <openrave/planningutils.h>
#include <ros/package.h>
#ifdef USE_OMPL
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/prm/PRM.h>
#endif
#include "rgbd_util/assert.h"
#include "rgbd_util/parallelism.h"
#include "xforms/xforms.h"
#include "pcl_rgbd/cloudSearchTrees.h"
#include "scene_rendering/triangulatedMeshRenderer.h"
#include "scene_rendering/castRaysIntoSurfels.h"
#include "rgbd_frame_common/rgbdFrame.h"
#include "vrip_utils/voxelGridDistanceTransforms.h"
#include "peter_intel_mapping_utils/rendering.h"
#include "peter_intel_mapping_utils/conversions.h"
#include "openrave_utils/openraveUtils.h"
#include "active_vision_common/openraveUtils.h"
#include "active_vision_common/simulatedScene.h"
#ifdef PLAN_WITH_CHOMP
extern "C"
{
#include "orcdchomp/libcd/mat.h"
#include "orcdchomp/libcd/grid.h"
#include "orcdchomp/libcd/kin.h"
}
#include "orcdchomp/orcdchompSDF.h"
#endif
#ifdef USE_SPHERE_CARVING
#include "active_obj_modeling/polyhedronApproximationWithSpheres.h"
#endif
#ifdef USE_APPROX_ROBOT_MESHES_FOR_COLLISION_CHECKING
#include "active_obj_modeling/robotMeshSimplification.h"
#endif
#include "active_obj_modeling/collisionChecking.h"
#include "active_obj_modeling/onlineActiveModeler.h"
using std::vector;
using std::ifstream;
using std::ofstream;
using std::cout;
using std::endl;
using boost::lexical_cast;
#ifdef USE_OMPL
namespace omplb = ompl::base;
namespace omplg = ompl::geometric;
#endif
using rgbd::eigen::Vector3f;

/************************************************** for debugging 20140515 ******************************************************/

int parseLine(char* line)
{
  int i = strlen(line);
  while (*line < '0' || *line > '9') line++;
  line[i-3] = '\0';
  i = atoi(line);
  return i;
}


int getMemUsageMB()
{
  FILE* file = fopen("/proc/self/status", "r");
  int result = -1;
  char line[128];

#if 1 //physical mem
  while (fgets(line, 128, file) != NULL){
	  if (strncmp(line, "VmRSS:", 6) == 0){
			result = parseLine(line);
			break;
	  }
	}
#else //virtual mem
  while (fgets(line, 128, file) != NULL){
		if (strncmp(line, "VmSize:", 7) == 0){
			 result = parseLine(line);
			 break;
		}
  }
#endif

  fclose(file);

  return result / 1000;
}

void reportMemUsage(const std::string& desc)
{
	cout << "[mem] vmem usage in MB, " << desc << ": " << getMemUsageMB() << endl;
}

/********************************************************************************************************************************/

triangulatedMesh makeOctahedronMesh(const rgbd::eigen::Vector3f& centroid, const float dx, const rgbd::eigen::Vector3f& rgb)
{
	triangulatedMesh mesh;
	mesh.allocateVertices(6);
	mesh.allocateTriangles(8);
	rgbd::pt pt;
	pt.rgb = rgbd::packRGB(rgb);
	pt.x = centroid.x() - dx; pt.y = centroid.y(); pt.z = centroid.z();
	mesh.v(0) = pt;
	pt.x = centroid.x() + dx; pt.y = centroid.y(); pt.z = centroid.z();
	mesh.v(1) = pt;
	pt.x = centroid.x(); pt.y = centroid.y() - dx; pt.z = centroid.z();
	mesh.v(2) = pt;
	pt.x = centroid.x(); pt.y = centroid.y() + dx; pt.z = centroid.z();
	mesh.v(3) = pt;
	pt.x = centroid.x(); pt.y = centroid.y(); pt.z = centroid.z() - dx;
	mesh.v(4) = pt;
	pt.x = centroid.x(); pt.y = centroid.y(); pt.z = centroid.z() + dx;
	mesh.v(5) = pt;
	mesh.setTriangle(0, triangulatedMesh::triangle{{{0, 4, 2}}}); //-x, -y, -z
	mesh.setTriangle(1, triangulatedMesh::triangle{{{0, 2, 5}}}); //-x, -y, +z
	mesh.setTriangle(2, triangulatedMesh::triangle{{{0, 3, 4}}}); //-x, +y, -z
	mesh.setTriangle(3, triangulatedMesh::triangle{{{0, 5, 3}}}); //-x, +y, +z
	mesh.setTriangle(4, triangulatedMesh::triangle{{{1, 4, 2}}}); //+x, +y, -z
	mesh.setTriangle(5, triangulatedMesh::triangle{{{1, 2, 5}}}); //+x, +y, +z
	mesh.setTriangle(6, triangulatedMesh::triangle{{{1, 3, 4}}}); //+x, -y, -z
	mesh.setTriangle(7, triangulatedMesh::triangle{{{1, 5, 3}}}); //+x, -y, +z
	return mesh;
}

/********************************************************************************************************************************************************************************************************************************/

onlineActiveModeler::triValueFuncType onlineActiveModeler::str2triValueFunc(const std::string& s) const
{
	if(s == "random_frontier") return onlineActiveModeler::triValueFuncType::RANDOM_FRONTIER;
	if(s == "distance_to_cam") return onlineActiveModeler::triValueFuncType::DISTANCE_TO_CAM;
	if(s == "distance_to_cur_target") return onlineActiveModeler::triValueFuncType::DISTANCE_TO_CUR_TARGET;
	if(s == "distance_to_moved_surface") return onlineActiveModeler::triValueFuncType::DISTANCE_TO_MOVED_SURFACE;
	if(s == "distance_hybrid_target_moved") return onlineActiveModeler::triValueFuncType::DISTANCE_HYBRID_TARGET_MOVED;
	if(s == "distance_hybrid_cam_surface") return onlineActiveModeler::triValueFuncType::DISTANCE_HYBRID_CAM_SURFACE;
	if(s == "distance_hybrid_target_surface") return onlineActiveModeler::triValueFuncType::DISTANCE_HYBRID_TARGET_SURFACE;
	if(s == "info_gain") return onlineActiveModeler::triValueFuncType::INFO_GAIN;
	if(s == "traversal_cost_frontier") return onlineActiveModeler::triValueFuncType::TRAVERSAL_COST_FRONTIER;
	if(s == "infogain_traversalcost") return onlineActiveModeler::triValueFuncType::DISTANCE_HYBRID_INFOGAIN_TRAVERSALCOST;
	ASSERT_ALWAYS(false && "bad string");
}

/*
 * we might edit differParams before we pass it along
 *
 * renderer must have had a large texture allocated, as we may do view scoring
 *
 * pre: camHandler->initROSSubscribers() has been called
 */
onlineActiveModeler::onlineActiveModeler(const std::shared_ptr<openglContext>& ctx, const std::shared_ptr<viewScoringRenderer>& renderer, const std::shared_ptr<robotSpec>& rspec, const std::shared_ptr<cameraOnRobotHandler>& camHandler,
	const std::shared_ptr<continuousTrajSender>& trajSender, const std::shared_ptr<onlineSceneDifferencer>& m, const configOptions& cfg, OpenRAVE::EnvironmentBasePtr visEnv, const rgbd::eigen::Affine3f& mapPoseWrtRaveWorld)
: glContext(ctx), sceneRenderer(renderer), robotInterface(rspec), onlineMapper(m), visEnv(visEnv), trajSender(trajSender)
{
	ASSERT_ALWAYS(camHandler);
	robotCamHandler = camHandler;
	cams = onlineMapper->getCameraSetup();
	const rgbd::CameraParams camParams(primesensor::getColorCamParams(cams.cam));

	outdir = cfg.get<std::string>("outdir");

	inSimulation = cfg.get<bool>("simulateRobot");

	triValueFunc = str2triValueFunc(cfg.get<std::string>("triValueFunc"));//triValueFuncType::TRAVERSAL_COST_FRONTIER;//DISTANCE_HYBRID_TARGET_SURFACE;

	const OpenRAVE::RobotBasePtr robot = visEnv->GetRobot(robotCamHandler->getRobotName());
	ASSERT_ALWAYS(robot);

	//educated guess at how long we'll take to run one mapping outer loop iter; TODO continually refresh during runtime w/ max taken so far, or something conservative like that?
	if(inSimulation)
		estimatedMappingIterationRuntime = (onlineMapper->getCurSceneMapParams().volume_modeler.model_type == MODEL_GRID) ? 2.5 : .2/* when the online differ actually isn't diffing or updating the bkgnd map; TODO parameterize! */;//.9;
	else
		estimatedMappingIterationRuntime = (onlineMapper->getCurSceneMapParams().volume_modeler.model_type == MODEL_GRID) ? 4 : .7;

	/*
	 * assuming the map volume is big enough, carve out space around the robot that we'll start by assuming is free, so the robot can make any movements at all
	 */
	const std::string robotName = cfg.get<std::string>("robotName");

	const VolumeModelerAllParams curMapParams = onlineMapper->getCurSceneMapParams();

	/*
	 * initialize info about locations we aren't able to add to the map for whatever reason
	 */
	viewsByPose.reset(curMapParams.volume.cell_size, curMapParams.volume.cell_count, (curMapParams.volume_modeler.model_type != MODEL_SINGLE_VOLUME), std::array<uint16_t, 6>{{0, 0, 0, 0, 0, 0}});

	diffingResultWrtCurMap.reset(curMapParams.volume.cell_size, curMapParams.volume.cell_count, (curMapParams.volume_modeler.model_type != MODEL_SINGLE_VOLUME));

{
	const std::vector<rgbd::eigen::Affine3f> boxPosesAndSizesInRobotCoords = robotInterface->getInitialFreeSpaceBoxes(); //for peter's code: scale gives size (full width in each dim); translation gives minimum corner; rotation can also be given
#ifdef USE_SPHERE_CARVING //not needed here now that Peter's given us setValueInBox(), but keep in case wanted for other shapes
	/*
	 * as of 20131029 we can carve space in a peter-intel map with spheres, but not with any other shape; so let's approximate other shapes with spheres
	 */
	rgbd::timer t;
	for(const rgbd::eigen::Affine3f& boxPoseWrtRobot : boxPosesAndSizesInRobotCoords)
	{
		TODO if I ever use this code again, fix lower and upper bounds using boxPoseWrtRobot
		std::vector<sphereInfo> spheres = approximateAABBWithSpheres(rgbd::eigen::Vector3f(-.8, -.6, -1)/* lower bounds */, rgbd::eigen::Vector3f(.15, .4, 1.1)/* upper bounds */, "initialRobotSpace");
		for(const sphereInfo& s : spheres)
		{
			//add the bbox center because the sphere positions seem to be centered around zero but do have about the right extent
			curSceneMapper->setValueInSphere(10/* tsdf value */, .1/* depth weight */, robotBasePoseWrtMap * s.c/* center */, s.r/* radius */);
		}
	}
	t.stop("fit spheres to bboxes");
#else
{
	rgbd::timer t;
	std::mutex& curMapMux = onlineMapper->getCurSceneMutex();
	std::lock_guard<std::mutex> lock(curMapMux);
	const std::shared_ptr<VolumeModeler> curSceneMapper = onlineMapper->getCurSceneModeler();
	for(const rgbd::eigen::Affine3f& boxPoseWrtRobot : boxPosesAndSizesInRobotCoords)
	{
		const rgbd::eigen::Affine3f boxPoseInMapCoords = mapPoseWrtRaveWorld.inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(robotCamHandler->getLatestConfiguration()) * boxPoseWrtRobot;
		curSceneMapper->setValueInBox(10/* tsdf value */, .1/* depth weight */, boxPoseInMapCoords);
	}
	t.stop("carve space around robot");
}
#endif
}

	curMapPoseWrtRaveWorld = mapPoseWrtRaveWorld;
	cout << "curMapPoseWrtRaveWorld:" << endl << curMapPoseWrtRaveWorld.matrix() << endl;

#ifdef USE_APPROX_ROBOT_MESHES_FOR_COLLISION_CHECKING
	/*
	 * create simplified versions of robot link meshes, for fast approximate collision checking
	 */
{
	cout << "simplifying robot meshes for approx collision checking" << endl;
	robotMeshSimplificationParams params;
	if(robotInterface->getRobotName() == "BarrettWAM") params.triDimThreshold = 7e-3;
	else if(robotInterface->getRobotName() == "FreeFlyingCamera") params.triDimThreshold = 1e-7;
	else ASSERT_ALWAYS(false && "unhandled robot");
	std::tie(simplifiedRobotLinkMeshes, robotLinkMeshLinkIndices) = std::move(createSimplifiedRobotMeshes(robot, params));
}
#else
	ASSERT_ALWAYS(false && "to be implemented: copy non-approx rave meshes directly into our structures");
#endif

{
	const std::vector<std::string> staticObstacleKinbodyNames = robotInterface->getStaticObstacleKinbodyNames(); //stuff other than the map mesh to be collision-checked
	raveEnvLock envLock(visEnv, "visEnv in activeModeler ctor"); // lock environment
	for(const std::string& name : staticObstacleKinbodyNames)
	{
		const OpenRAVE::KinBodyPtr bodyptr = visEnv->GetKinBody(name);
		ASSERT_ALWAYS(bodyptr);
		const std::shared_ptr<triangulatedMesh> linkMesh = raveBody2trimesh(bodyptr);
		staticRaveEnvMeshes.push_back(linkMesh);
	}
}

	/*
	 * make openrave environment copies for motion planning
	 */
{
	planningEnv = visEnv->CloneSelf(OpenRAVE::Clone_Bodies);
	planningEnv->SetCollisionChecker(OpenRAVE::RaveCreateCollisionChecker(planningEnv, "ode")); //the environment used by chomp needs to have a collision checker if chomp is going to make any of its own SDFs

	const size_t numThreads = getSuggestedThreadCount(2, 2);
	raveEnvCopies.resize(numThreads);
	for(size_t i = 0; i < numThreads; i++)
	{
		raveEnvCopies[i] = visEnv->CloneSelf(OpenRAVE::Clone_Bodies);
		raveEnvCopies[i]->SetCollisionChecker(OpenRAVE::RaveCreateCollisionChecker(raveEnvCopies[i], "ode")); //so we can do robot-robot collision checking
	}
}

#ifdef PLAN_WITH_CHOMP
	chompModule = OpenRAVE::RaveCreateModule(planningEnv, "orcdchomp"); //orcdchomp is the openrave CHOMP module; this is Arun's edited version (I do depend on that fact; I've also edited the code)
	ASSERT_ALWAYS(chompModule);
	planningEnv->AddModule(chompModule, ""/* cmd-line args */);
#endif

	moveRobot = cfg.get<bool>("runOnRobot");
	if(moveRobot)
	{
		resetPlanStartConfig = true; //make sure that the first time we want to plan we use valid dof values rather than something uninitialized
		trajSender->addTrajCompletionCallback([this](const int64_t planID, const bool success)
			{
				if(success)
				{
					std::lock_guard<std::mutex> lock(plansCompletedMux);
					plansCompleted.insert(planID);
				}
				else
				{
					std::lock_guard<std::mutex> lock(planStartConfigMux);
					resetPlanStartConfig = true;
					//TODO abandon the current target pose if a traj failed to run?
				}
			});

		trajSenderThread.reset(new std::thread([this]()
			{
				this->trajSender->run();
			}));
	}

	/*
	 * runtime statistics
	 */
	nextTargetIDToUse = 0;
	lastTargetReached = -1;
	nextPlanIDToUse = 0;
	experimentOver = false;
	distanceTraveled = 0;
	numPlans = numSuccessfulPlans = 0;
	numPoseSuggestionsComputed = numPoseSuggestionsDiscarded = 0;
	lastTargetReachedByExistingPlan = -1;
	totalUpdateTime = 0;
	updateCount = 0;
	trackKnownFreeSpace = true; //TODO ?

	if(trackKnownFreeSpace)
	{
		OpenRAVE::EnvironmentBasePtr tmpEnv = OpenRAVE::RaveCreateEnvironment();
		simulatedScene simScene; //TODO make sure this matches what's used in the driver file
		simScene.addToEnv(tmpEnv);
		const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();
		const rgbd::eigen::Affine3f camPoseWrtRaveWorld = robotCamHandler->getRobotBasePoseWrtRaveWorld(curConfig) * robotCamHandler->getCamPoseWrtRobotBase(curConfig);
		const auto& curMapParams = onlineMapper->getCurSceneMapParams();
		initialSceneFreeVoxels.voxels.resize(boost::extents[curMapParams.volume.cell_count[2]][curMapParams.volume.cell_count[1]][curMapParams.volume.cell_count[0]]);
		size_t initialNumFreeVoxels;
		std::tie(initialSceneFreeVoxels, initialNumFreeVoxels) = std::move(simScene.numReachableFreeVoxels(onlineMapper->getCurSceneMapParams(), curMapPoseWrtRaveWorld, camPoseWrtRaveWorld, tmpEnv));
		cout << "sim scene: " << initialNumFreeVoxels << " free voxels at start" << endl;
	}

	clock_gettime(CLOCK_REALTIME, &modelingStartTime);
}

cv::Mat_<cv::Vec3b> onlineActiveModeler::getTargetCamPoseImg() const
{
	return targetCamPoseImg;
}

std::string onlineActiveModeler::getTargetCamPoseDescription() const
{
	std::ostringstream outstr;

	if(curTarget)
	{
		if(curTarget->targetInfo.triIndex >= 0)
		{
			const rgbd::eigen::Vector3f x0 = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curTarget->targetInfo.triVertices[0]),
				x1 = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curTarget->targetInfo.triVertices[1]),
				x2 = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curTarget->targetInfo.triVertices[2]);
			const float triArea = .5 * (x1 - x0).cross(x2 - x0).norm();
			outstr << "area " << triArea;

			const rgbd::eigen::Vector3f triNormal = (x1 - x0).cross(x2 - x0).normalized();
			outstr << "\nnormal " << triNormal.transpose();

#if 0 //the data race here rarely happens except in valgrind, but does keep happening there
			const rgbd::eigen::Vector3f tgtPosInRaveWorld = curMapPoseWrtRaveWorld * curTarget->camPoseWrtMap.get() * rgbd::eigen::Vector3f(0, 0, .4/* = viewDist; TODO ? */);
			const float tsdfVal = tsdfValueAtPos(tgtPosInRaveWorld);
			outstr << "\ntsdf val " << tsdfVal;
#endif
		}
		else outstr << "[not viewing a tri]";
	}
	else outstr << "[no cur target]";

	return outstr.str();
}

/*
 * return some gui-able info about the last time we abandoned a target
 */
std::string onlineActiveModeler::getTargetSwitchInfo() const
{
	return targetSwitchStateStr;
}

cv::Mat_<cv::Vec3b> onlineActiveModeler::getMeshValuesImg() const
{
	return meshValuesImg;
}

/*
 * do we use said info for anything? if not, don't update the per-voxel diffing-results map each frame
 */
bool onlineActiveModeler::shouldKeepDiffingInfoForCurScene() const
{
#ifdef RUN_DIFFING_SCORING_EXPERIMENT
	return true;
#else
	return triValueFunc == triValueFuncType::DISTANCE_TO_MOVED_SURFACE || triValueFunc == triValueFuncType::DISTANCE_HYBRID_TARGET_MOVED;
#endif
}

/*
 * update synchronously with modeling, after each frame of modeling
 * (eg, so we can copy results from this frame that the mapper won't store for future frames)
 */
void onlineActiveModeler::updatePerModelingFrame(rgbdFrame& frame, const boost::posix_time::ptime& frameTime)
{
	//don't need to hold any mutices since modeling isn't writing right now

	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);

	const size_t frameIndex = onlineMapper->getLastFrameIndex();

	framePosesWrtCurMap.resize(frameIndex + 1);
	framePosesWrtCurMap[frameIndex] = onlineMapper->getFramePoseWrtCurMap(frameIndex);

	if(shouldKeepDiffingInfoForCurScene())
	{
		/*
		 * aggregate diffing info into voxels representing the cur-scene map
		 *
		 * (if we don't do it in this function, we have to either (a) keep an unbounded buffer of these things per frame around and update with a bunch of frames at once each time we updateAsync
		 * or (b) always keep just the last frame's worth of this info around and update with only the most recent frame in each updateAsync)
		 */

		rgbd::timer t;

		const VolumeModelerAllParams& curSceneParams = onlineMapper->getCurSceneMapParams();
		const float curMapVoxelSize = curSceneParams.volume.cell_size;
		const rgbd::eigen::Affine3f latestCamPoseWrtCurMap = onlineMapper->getCurMapPoseWrtBkgndMap().inverse() * onlineMapper->getFramePoseWrtBkgndMap(frameIndex);

		const cv::Mat_<float>& latestDepth = frame.getDepthImg();
		const boost::multi_array<float, 2>& latestDepthStdev = frame.getDepthUncertainty();
		const sumLogprobsCombined latestDiffingResultWrtPrevScene(onlineMapper->getLastFrameLogprobSumsWrtMap());
		diffingResultWrtCurMap.runForEachVoxelInCameraFrustum(camParams, latestCamPoseWrtCurMap,
			[&](const int64_t ix, const int64_t iy, const int64_t iz, const float u, const float v, const float z)
				{
					const int iu = clamp((int)rint(u), 0, (int)camParams.xRes - 1), iv = clamp((int)rint(v), 0, (int)camParams.yRes - 1);
					//if there was a valid frame depth and this voxel's close to that depth according to the frame depth stdev, add diffing info to this voxel
					if(latestDepth(iv, iu) > 0 && fabs(z - latestDepth(iv, iu)) < std::max(curMapVoxelSize * .7, latestDepthStdev[iv][iu] * 2.5/* TODO ? */))
					{
						voxelDiffingInfo& info = diffingResultWrtCurMap(std::array<int64_t, 3>{{ix, iy, iz}});
						info.evidenceMoved += latestDiffingResultWrtPrevScene.logprobsGivenMoved[iv * camParams.xRes + iu];
						info.evidenceNotMoved += latestDiffingResultWrtPrevScene.logprobsGivenNotMoved[iv * camParams.xRes + iu];
					}
				});

		t.stop("add diffing info to voxels");
	}
}

/*
 * scores in [0, 1], 1 best
 *
 * if an outpath is empty, don't write the corresponding file
 */
cv::Mat_<cv::Vec3b> onlineActiveModeler::visualizeMeshWithCameraPoses(const triangulatedMesh& mesh, const std::vector<std::pair<rgbd::eigen::Affine3f, float>>& camPosesWrtMapWithScores,
	const rgbd::eigen::Affine3f& viewingPoseWrtMap, const bool showCurCamPose, const fs::path& imgOutpath, const fs::path& meshOutpath) const
{
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);

	std::shared_ptr<triangulatedMesh> curMapMeshPlus(new triangulatedMesh(mesh));
	curMapMeshPlus->allocateVertices(mesh.numVertices() + 5 * (camPosesWrtMapWithScores.size() + (showCurCamPose ? 1 : 0)));
	curMapMeshPlus->allocateTriangles(mesh.numTriangles() + 8 * (camPosesWrtMapWithScores.size() + (showCurCamPose ? 1 : 0)));

	const auto addCamPose = [&](const rgbd::eigen::Affine3f& camPoseWrtMap, const float score, const size_t frustumIndex)
		{
			const auto camPoseWrtMesh = camPoseWrtMap;
#define N (uint32_t)(mesh.numVertices() + frustumIndex * 5)
#define M (mesh.numTriangles() + frustumIndex * 8)
			rgbd::pt x[5];
			rgbd::eigen2ptX<rgbd::pt>(x[0], rgbd::eigen::Vector3f(camPoseWrtMesh * rgbd::eigen::Vector3f(0, 0, 0)));
			rgbd::eigen2ptX<rgbd::pt>(x[1], rgbd::eigen::Vector3f(camPoseWrtMesh * rgbd::eigen::Vector3f(-.02, -.02, .06)));
			rgbd::eigen2ptX<rgbd::pt>(x[2], rgbd::eigen::Vector3f(camPoseWrtMesh * rgbd::eigen::Vector3f(-.02, .02, .06)));
			rgbd::eigen2ptX<rgbd::pt>(x[3], rgbd::eigen::Vector3f(camPoseWrtMesh * rgbd::eigen::Vector3f(.02, .02, .06)));
			rgbd::eigen2ptX<rgbd::pt>(x[4], rgbd::eigen::Vector3f(camPoseWrtMesh * rgbd::eigen::Vector3f(.02, -.02, .06)));
			if(score == -1) //special coloring
			{
				x[0].rgb = rgbd::packRGB(boost::array<uint8_t, 3>{{255, 0, 0}});
				x[1].rgb = rgbd::packRGB(boost::array<uint8_t, 3>{{255, 0, 0}});
				x[2].rgb = rgbd::packRGB(boost::array<uint8_t, 3>{{255, 128, 0}});
				x[3].rgb = rgbd::packRGB(boost::array<uint8_t, 3>{{255, 0, 0}});
				x[4].rgb = rgbd::packRGB(boost::array<uint8_t, 3>{{255, 128, 0}});
			}
			else //continuous coloring
			{
				x[0].rgb = rgbd::packRGB(boost::array<uint8_t, 3>{{255, 255, 0}});
				x[1].rgb = rgbd::packRGB(boost::array<uint8_t, 3>{{(uint8_t)(255 * score), 255, 0}});
				x[2].rgb = rgbd::packRGB(boost::array<uint8_t, 3>{{(uint8_t)(255 * score), 128, 0}});
				x[3].rgb = rgbd::packRGB(boost::array<uint8_t, 3>{{(uint8_t)(255 * score), 255, 0}});
				x[4].rgb = rgbd::packRGB(boost::array<uint8_t, 3>{{(uint8_t)(255 * score), 128, 0}});
			}
			for(int q = 0; q < 5; q++) curMapMeshPlus->setVertex(q + N, x[q]);
			curMapMeshPlus->setTriangle(M+0, triangulatedMesh::triangle{{{0+N, 1+N, 2+N}}});
			curMapMeshPlus->setTriangle(M+1, triangulatedMesh::triangle{{{0+N, 2+N, 3+N}}});
			curMapMeshPlus->setTriangle(M+2, triangulatedMesh::triangle{{{0+N, 3+N, 4+N}}});
			curMapMeshPlus->setTriangle(M+3, triangulatedMesh::triangle{{{0+N, 4+N, 1+N}}});
			curMapMeshPlus->setTriangle(M+4, triangulatedMesh::triangle{{{0+N, 2+N, 1+N}}});
			curMapMeshPlus->setTriangle(M+5, triangulatedMesh::triangle{{{0+N, 3+N, 2+N}}});
			curMapMeshPlus->setTriangle(M+6, triangulatedMesh::triangle{{{0+N, 4+N, 3+N}}});
			curMapMeshPlus->setTriangle(M+7, triangulatedMesh::triangle{{{0+N, 1+N, 4+N}}});
#undef N
#undef M
		};

	for(size_t p = 0; p < camPosesWrtMapWithScores.size(); p++)
	{
		const auto camPoseWrtMesh = camPosesWrtMapWithScores[p].first;
		float dx, da;
		xf::transform_difference(rgbd::eigen::Affine3f::Identity(), camPoseWrtMesh, dx, da);
		const float score = camPosesWrtMapWithScores[p].second; //in [0, 1], for coloring
		addCamPose(camPosesWrtMapWithScores[p].first, score, p);
	}
	if(showCurCamPose)
	{
		ASSERT_ALWAYS(framePosesWrtCurMap.size() > frameIndex);
	//	cout << "cur frame pose:" << endl << framePosesWrtCurMap[frameIndex].matrix() << endl;
		addCamPose(framePosesWrtCurMap[frameIndex], -1, camPosesWrtMapWithScores.size());
	}

	cv::Mat_<cv::Vec3b> bgrImg(camParams.yRes, camParams.xRes);
#if 1//def DEBUGGING_20140516
	glContext->acquire();
{
	const triangulatedMeshRenderer::vertexColoringFunc coloringFunc = getMeshVertexColorFromPointColor;
	std::shared_ptr<triangulatedMeshRenderer> tmpRenderer(new triangulatedMeshRenderer(*curMapMeshPlus, coloringFunc, camParams));
	sceneRenderer->acquire();
	sceneRenderer->setRenderFunc([&tmpRenderer](const rgbd::eigen::Affine3f& camPose) {tmpRenderer->render(camPose);});
	sceneRenderer->render(viewingPoseWrtMap, bgrImg);
	sceneRenderer->restoreRenderFunc();
	sceneRenderer->release();
} //ensure the triangulatedMeshRenderer releases its opengl resources while its context is active
	glContext->release();
	if(!imgOutpath.empty()) cv::imwrite(imgOutpath.string(), bgrImg);
#endif
	if(!meshOutpath.empty()) curMapMeshPlus->writePLY(meshOutpath);
	return bgrImg;
}

/*
 * visualize robot + environment as meshes
 */
void onlineActiveModeler::visualizeRobotPoses(const std::shared_ptr<triangulatedMesh>& envMesh, const std::vector<std::vector<OpenRAVE::dReal>>& configurations, robotSpec& robotInterface, const fs::path& outdirPlusFilebase) const
{
	for(size_t i = 0; i < configurations.size(); i++)
		if(!configurations[i].empty())
		{
			std::vector<OpenRAVE::Transform> raveLinkPosesWrtRaveWorld;
			OpenRAVE::RobotBasePtr robot = raveEnvCopies[0]->GetRobot(robotCamHandler->getRobotName());
			robotInterface.setRAVERobotConfiguration(robot, configurations[i]);
			robot->GetLinkTransformations(raveLinkPosesWrtRaveWorld);

			triangulatedMesh fullMesh;
			triangulatedMesh xformedEnvMesh = *envMesh;
			for(size_t k = 0; k < xformedEnvMesh.numVertices(); k++)
				rgbd::eigen2ptX(xformedEnvMesh.v(k), curMapPoseWrtRaveWorld * rgbd::ptX2eigen<rgbd::eigen::Vector3f>(xformedEnvMesh.v(k)));
			fullMesh.append(xformedEnvMesh);
			for(const auto& m : staticRaveEnvMeshes) fullMesh.append(*m);
			for(size_t j = 0; j < simplifiedRobotLinkMeshes.size(); j++)
			{
				triangulatedMesh xformedMesh = simplifiedRobotLinkMeshes[j];
				for(size_t k = 0; k < xformedMesh.numVertices(); k++)
				{
					rgbd::eigen2ptX(xformedMesh.v(k), raveXform2eigenXform(raveLinkPosesWrtRaveWorld[robotLinkMeshLinkIndices[j]]) * rgbd::ptX2eigen<rgbd::eigen::Vector3f>(xformedMesh.v(k)));
					xformedMesh.v(k).rgb = rgbd::packRGB(192, 192, 192); //it'll be black and very hard to see if we don't
				}
				fullMesh.append(xformedMesh);
			}

			const fs::path outpath = outdirPlusFilebase.string() + boost::lexical_cast<std::string>(i) + ".ply";
			fullMesh.writePLY(outpath);
		}
}

/*************************************************************************************************************************************************************************************************************************************/

/*
 * auxiliary to asynchronous update
 *
 * extract all the data structures we need from maps for this iteration
 */
void onlineActiveModeler::processMapsBeforeViewSelection()
{
	rgbd::timer t, t2;
	t.restart();
	frameIndex = onlineMapper->getLastFrameIndex();
	t.stop("clear mapping info buffers");
	reportMemUsage("begin processMaps");

	t.restart();
	/*
	 * here's the one place per planning run where we get a copy of the map; all other info about the map is derived from the variables set here
	 */
{
	std::mutex& curMapMux = onlineMapper->getCurSceneMutex();
	const std::shared_ptr<VolumeModeler> curSceneMapper = onlineMapper->getCurSceneModeler();
	rgbd::timer t;
{
	std::lock_guard<std::mutex> lock(curMapMux);
	curSceneMapper->getTSDFBuffersLists(bufferDVectors, bufferDWeightVectors, bufferCVectors, bufferCWeightVectors, tsdfBufferPosesWrtMap, bufferVoxelCounts);
}
	t.stop("get tsdf buffers");
	t.restart();
	//the operation of getting a mesh doesn't require synchronization if we already have the tsdf buffers
	std::tie(curMapMesh, triangleIsSurface) = std::move(generateIntelMapMeshWithUnseenness(*curSceneMapper, bufferDVectors, bufferDWeightVectors, bufferCVectors, bufferCWeightVectors, bufferVoxelCounts, tsdfBufferPosesWrtMap)); //very slow (> 1s for a 256^3 single volume that's not almost empty)
	t.stop("get cur map mesh");
	cout << "cur mesh has " << curMapMesh->numTriangles() << " triangles" << endl;
}
	t.stop("get cur map info");

#if 1 //debugging 20140227; TODO remove once sure this doesn't break (but it does once in a while! and I ought to be able to use a freeness threshold of at least one voxel's worth -- ??)
	const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();
	const rgbd::eigen::Affine3f curCamPoseWrtRaveWorld = robotCamHandler->getRobotBasePoseWrtRaveWorld(curConfig) * robotCamHandler->getCamPoseWrtRobotBase(curConfig);
	if(tsdfValueAtPos(curCamPoseWrtRaveWorld * rgbd::eigen::Vector3f(0, 0, 0)) < .001/* onlineMapper->getCurSceneMapParams().volume.cell_size * 1/* TODO ? */)
	{
		cout << "tsdf no longer free at cam pos: tsdf at " << (curCamPoseWrtRaveWorld * rgbd::eigen::Vector3f(0, 0, 0)).transpose() << " is " << tsdfValueAtPos(curCamPoseWrtRaveWorld * rgbd::eigen::Vector3f(0, 0, 0)) << endl;
		raise(SIGINT); //stop the debugger
	}
#endif

	/*
	 * extract the seen parts of the mesh (very fast)
	 */
	t.restart();
	curMapMeshSeenOnly.reset(new triangulatedMesh);
{
	size_t numSeenTris = 0;
	for(uint8_t s : triangleIsSurface)
		if(s)
			numSeenTris++;
	curMapMeshSeenOnly->allocateVertices(curMapMesh->numVertices());
	curMapMeshSeenOnly->allocateTriangles(numSeenTris);
	for(size_t q = 0; q < curMapMesh->numVertices(); q++) curMapMeshSeenOnly->setVertex(q, curMapMesh->v(q));
	const auto& tris = curMapMesh->getTriangles();
	for(size_t q = 0, qs = 0; q < tris.size(); q++)
		if(triangleIsSurface[q])
			curMapMeshSeenOnly->setTriangle(qs++, tris[q]);
}
	const size_t numUnseenTriangles = curMapMesh->numTriangles() - curMapMeshSeenOnly->numTriangles();
	cout << "cur map has " << numUnseenTriangles << " unseen triangles" << endl;
	if(numUnseenTriangles == 0) experimentOver = true;
	t.stop("extract seen mesh");

	/*
	 * put the mesh into openrave for visualization and maybe collision checking and maybe simulated sensing
	 */
{
	rgbd::timer t;
	const std::shared_ptr<triangulatedMesh> curMeshForOpenrave = curMapMesh;
	OpenRAVE::TriMesh raveMesh;
	triangulatedMesh2openraveMesh(*curMeshForOpenrave, curMapPoseWrtRaveWorld, raveMesh);
	t.stop("copy mesh structure to rave");

	/*
	 * add the entire mesh in one piece (so it's all the same color)
	 */
	const auto addMeshToEnv = [&raveMesh](const OpenRAVE::EnvironmentBasePtr& e)
		{
			OpenRAVE::KinBodyPtr raveMeshPtr = OpenRAVE::RaveCreateKinBody(e); //it won't work if I ask for a non-empty name here
			ASSERT_ALWAYS(raveMeshPtr);
			raveMeshPtr->InitFromTrimesh(raveMesh, true/* visible */);
			raveMeshPtr->SetName("mapMesh"); //if I don't set a name, AddKinBody() will fail
			setRaveBodyColor(raveMeshPtr, rgbd::eigen::Vector3f(1, 0, .6), .6);

			raveEnvLock envLock(e, "addMeshToEnv");
			const auto bodyptr = e->GetKinBody("mapMesh");
			if(bodyptr) e->Remove(bodyptr);
			e->Add(raveMeshPtr); //if I use the default openrave collision checker (ode) for this env, this call takes many seconds, almost all for adding the body to the collision checker
		};

	/*
	 * add seen and unseen triangles in different colors, for visualization
	 */
	const auto addSeenAndUnseenMeshesToEnv = [this](const OpenRAVE::EnvironmentBasePtr& e)
		{
			const auto& mesh = *curMapMesh;
			OpenRAVE::TriMesh raveMeshSeen, raveMeshUnseen;

			raveMeshSeen.vertices.resize(mesh.numVertices());
			raveMeshUnseen.vertices.resize(mesh.numVertices());
			for(uint32_t i = 0; i < mesh.numVertices(); i++) //put vertices into rave world coords
			{
				const rgbd::eigen::Vector4f ptInWorldCoords = curMapPoseWrtRaveWorld * rgbd::ptX2eigen<rgbd::eigen::Vector4f>(mesh.v(i));

				raveMeshSeen.vertices[i][0] = ptInWorldCoords.x();
				raveMeshSeen.vertices[i][1] = ptInWorldCoords.y();
				raveMeshSeen.vertices[i][2] = ptInWorldCoords.z();
				raveMeshSeen.vertices[i][3] = 0;

				raveMeshUnseen.vertices[i][0] = ptInWorldCoords.x();
				raveMeshUnseen.vertices[i][1] = ptInWorldCoords.y();
				raveMeshUnseen.vertices[i][2] = ptInWorldCoords.z();
				raveMeshUnseen.vertices[i][3] = 0;
			}

			raveMeshSeen.indices.resize(curMapMeshSeenOnly->numTriangles() * 3);
			raveMeshUnseen.indices.resize((curMapMesh->numTriangles() - curMapMeshSeenOnly->numTriangles()) * 3);
			const vector<triangulatedMesh::triangle>& tris = mesh.getTriangles();
			for(uint32_t i = 0, is = 0, iu = 0; i < mesh.numTriangles(); i++)
				if(triangleIsSurface[i])
				{
					raveMeshSeen.indices[is * 3 + 0] = tris[i].v[0];
					raveMeshSeen.indices[is * 3 + 1] = tris[i].v[1];
					raveMeshSeen.indices[is * 3 + 2] = tris[i].v[2];
					is++;
				}
				else
				{
					raveMeshUnseen.indices[iu * 3 + 0] = tris[i].v[0];
					raveMeshUnseen.indices[iu * 3 + 1] = tris[i].v[1];
					raveMeshUnseen.indices[iu * 3 + 2] = tris[i].v[2];
					iu++;
				}

		{
			OpenRAVE::KinBodyPtr raveMeshPtr = OpenRAVE::RaveCreateKinBody(e); //it won't work if I ask for a non-empty name here
			ASSERT_ALWAYS(raveMeshPtr);
			raveMeshPtr->InitFromTrimesh(raveMeshSeen, true/* visible */);
			raveMeshPtr->SetName("mapMeshSeen"); //if I don't set a name, AddKinBody() will fail
			setRaveBodyColor(raveMeshPtr, rgbd::eigen::Vector3f(1, .7, .6), .2);

			raveEnvLock envLock(e, "addMeshToEnv");
			const auto bodyptr = e->GetKinBody("mapMeshSeen");
			if(bodyptr) e->Remove(bodyptr);
			e->Add(raveMeshPtr); //if I use the default openrave collision checker (ode) for this env, this call takes many seconds, almost all for adding the body to the collision checker
		}

		{
			OpenRAVE::KinBodyPtr raveMeshPtr = OpenRAVE::RaveCreateKinBody(e); //it won't work if I ask for a non-empty name here
			ASSERT_ALWAYS(raveMeshPtr);
			raveMeshPtr->InitFromTrimesh(raveMeshUnseen, true/* visible */);
			raveMeshPtr->SetName("mapMeshUnseen"); //if I don't set a name, AddKinBody() will fail
			setRaveBodyColor(raveMeshPtr, rgbd::eigen::Vector3f(.7, .9, 1), .75);

			raveEnvLock envLock(e, "addMeshToEnv");
			const auto bodyptr = e->GetKinBody("mapMeshUnseen");
			if(bodyptr) e->Remove(bodyptr);
			e->Add(raveMeshPtr); //if I use the default openrave collision checker (ode) for this env, this call takes many seconds, almost all for adding the body to the collision checker
		}

		};

	/*
	 * show the actual scene in one color, the previously seen bits in another, and the bits currently being seen in yet another
	 */
	const auto addThreePartMapMeshToEnv = [this](const OpenRAVE::EnvironmentBasePtr& visEnv)
		{
			ASSERT_ALWAYS(inSimulation); //this function isn't defined if we're not simulating the scene

			/*
			 * the whole scene
			 */
			static bool init = true;
			if(init)
			{
				simulatedScene simScene;
				simScene.addToEnv(visEnv);
				std::vector<OpenRAVE::KinBodyPtr> kinbodies;
				visEnv->GetBodies(kinbodies);
				for(const auto& b : kinbodies) setRaveBodyColor(b, rgbd::eigen::Vector3f(.7, .9, 1), .75);
				init = false;
			}

			/*
			 * surfaces currently being viewed, and previously viewed surfaces not currently being viewed
			 */
		{
			//render map mesh from cur cam pose
			const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
			const triangulatedMeshRenderer::triangleColoringFunc coloringFunc = getTriangleColorFromID;
			boost::multi_array<uint32_t, 2> sampleIDs(boost::extents[camParams.yRes][camParams.xRes]);
			boost::multi_array<float, 2> sampleDepths(boost::extents[camParams.yRes][camParams.xRes]);
			glContext->acquire();
		{
			std::shared_ptr<triangulatedMeshRenderer> meshRenderer(new triangulatedMeshRenderer(*curMapMesh, coloringFunc, camParams));
			sceneRenderer->acquire();
			sceneRenderer->setRenderFunc([&meshRenderer](const rgbd::eigen::Affine3f& camPose) {meshRenderer->render(camPose);});
			projectSceneSamplesIntoCamera(*sceneRenderer, camParams, framePosesWrtCurMap[frameIndex], sampleIDs, sampleDepths);
			sceneRenderer->restoreRenderFunc();
			sceneRenderer->release();
		} //ensure the triangulatedMeshRenderer releases its opengl resources while its context is active
			glContext->release();

			//list tris to add to meshes
			std::unordered_set<uint32_t> visibleTriIDs; //tris currently being viewed
			for(size_t i = 0; i < camParams.yRes; i++)
				for(size_t j = 0; j < camParams.xRes; j++)
					if(sampleIDs[i][j] > 0)
						if(triangleIsSurface[sampleIDs[i][j] - 1])
							visibleTriIDs.insert(sampleIDs[i][j] - 1);
			std::vector<size_t> previouslyViewedTriIDs; //previously viewed tris not currently being viewed
			for(size_t i = 0; i < curMapMesh->numTriangles(); i++)
				if(triangleIsSurface[i] && visibleTriIDs.find(i) == visibleTriIDs.end())
					previouslyViewedTriIDs.push_back(i);

			//make meshes
			triangulatedMesh seenSurfacesMesh;
			seenSurfacesMesh.allocateVertices(curMapMesh->numVertices());
			for(size_t i = 0; i < curMapMesh->numVertices(); i++) seenSurfacesMesh.setVertex(i, curMapMesh->v(i));
			seenSurfacesMesh.allocateTriangles(visibleTriIDs.size());
			const auto& tris = curMapMesh->getTriangles();
			size_t i = 0;
			for(uint32_t t : visibleTriIDs) seenSurfacesMesh.setTriangle(i++, tris[t]);
			triangulatedMesh prevSeenSurfacesMesh;
			prevSeenSurfacesMesh.allocateVertices(curMapMesh->numVertices());
			for(size_t i = 0; i < curMapMesh->numVertices(); i++) prevSeenSurfacesMesh.setVertex(i, curMapMesh->v(i));
			prevSeenSurfacesMesh.allocateTriangles(previouslyViewedTriIDs.size());
			for(size_t i = 0; i < previouslyViewedTriIDs.size(); i++) prevSeenSurfacesMesh.setTriangle(i, tris[previouslyViewedTriIDs[i]]);
			OpenRAVE::TriMesh raveMeshCurSeen, raveMeshPrevSeen;
			triangulatedMesh2openraveMesh(seenSurfacesMesh, curMapPoseWrtRaveWorld, raveMeshCurSeen);
			triangulatedMesh2openraveMesh(prevSeenSurfacesMesh, curMapPoseWrtRaveWorld, raveMeshPrevSeen);

			//add the meshes to the env
			OpenRAVE::KinBodyPtr raveMeshPtrCurSeen = OpenRAVE::RaveCreateKinBody(visEnv); //it won't work if I ask for a non-empty name here
			ASSERT_ALWAYS(raveMeshPtrCurSeen);
			raveMeshPtrCurSeen->InitFromTrimesh(raveMeshCurSeen, true/* visible */);
			raveMeshPtrCurSeen->SetName("mapMeshCurrentlySeen"); //if I don't set a name, AddKinBody() will fail
			setRaveBodyColor(raveMeshPtrCurSeen, rgbd::eigen::Vector3f(0, .6, 0), 0);
			OpenRAVE::KinBodyPtr raveMeshPtrPrevSeen = OpenRAVE::RaveCreateKinBody(visEnv); //it won't work if I ask for a non-empty name here
			ASSERT_ALWAYS(raveMeshPtrPrevSeen);
			raveMeshPtrPrevSeen->InitFromTrimesh(raveMeshPrevSeen, true/* visible */);
			raveMeshPtrPrevSeen->SetName("mapMeshPreviouslySeen"); //if I don't set a name, AddKinBody() will fail
			setRaveBodyColor(raveMeshPtrPrevSeen, rgbd::eigen::Vector3f(1, .5, 0), .25);

			raveEnvLock envLock(visEnv, "addMeshToEnv");
			auto bodyptr = visEnv->GetKinBody("mapMeshCurrentlySeen");
			if(bodyptr) visEnv->Remove(bodyptr);
			visEnv->Add(raveMeshPtrCurSeen); //if I use the default openrave collision checker (ode) for this env, this call takes many seconds, almost all for adding the body to the collision checker
			bodyptr = visEnv->GetKinBody("mapMeshPreviouslySeen");
			if(bodyptr) visEnv->Remove(bodyptr);
			visEnv->Add(raveMeshPtrPrevSeen); //if I use the default openrave collision checker (ode) for this env, this call takes many seconds, almost all for adding the body to the collision checker
		}

		};

	t.restart();
#if 0 //visualization showing what surfaces the camera sees now, but not showing frontiers
	addThreePartMapMeshToEnv(visEnv); //for visualization
#else //visualization showing frontiers, but showing seen surfaces all in one color
	addSeenAndUnseenMeshesToEnv(visEnv); //for visualization
#endif
#if !defined(MAKE_MAP_MESH_SDF_OURSELVES)
	addMeshToEnv(planningEnv); //so chomp can make an sdf of the mesh
#else
	/*
	 * for efficiency, avoid adding the mesh to rave
	 *
	 * but chomp still needs a kinbody by that name in order to get its pose wrt rave world, so add something by that name
	 */
{
	OpenRAVE::KinBodyPtr newBodyPtr = OpenRAVE::RaveCreateKinBody(planningEnv); //it won't work if I ask for a non-empty name here
	ASSERT_ALWAYS(newBodyPtr);
	std::vector<OpenRAVE::AABB> boxes;
	boxes.push_back(OpenRAVE::AABB(OpenRAVE::geometry::RaveVector<float>(1000, 1000, 1000)/* center - make it out of the way of the scene */, OpenRAVE::geometry::RaveVector<float>(1, 1, 1)/* half-widths */));
	newBodyPtr->InitFromBoxes(boxes, false/* visible */);
	newBodyPtr->SetName("mapMesh"); //if I don't set a name, AddKinBody() will fail

	raveEnvLock envLock(planningEnv, "addMeshToEnv");
	const auto bodyptr = planningEnv->GetKinBody("mapMesh");
	if(bodyptr) planningEnv->Remove(bodyptr);
	planningEnv->Add(newBodyPtr);
}
#endif
	t.stop("visualize scene in openrave");

	t.restart();
	if(true) //TODO ?
	{
		/*
		 * visualize camera frustum
		 */
		const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();
		const rgbd::eigen::Affine3f curCamPoseWrtMap = curMapPoseWrtRaveWorld.inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(curConfig) * robotCamHandler->getCamPoseWrtRobotBase(curConfig);
		triangulatedMesh frMesh;
		frMesh.allocateVertices(5);
		const float dx = 3 * tan(M_PI / 6); //for a 60-deg fov in each of x and y, which is close enough
		rgbd::eigen2ptX(frMesh.v(0), rgbd::eigen::Vector3f(curCamPoseWrtMap * rgbd::eigen::Vector3f(0, 0, 0)));
		rgbd::eigen2ptX(frMesh.v(1), rgbd::eigen::Vector3f(curCamPoseWrtMap * rgbd::eigen::Vector3f(-dx, dx, 3)));
		rgbd::eigen2ptX(frMesh.v(2), rgbd::eigen::Vector3f(curCamPoseWrtMap * rgbd::eigen::Vector3f(-dx, -dx, 3)));
		rgbd::eigen2ptX(frMesh.v(3), rgbd::eigen::Vector3f(curCamPoseWrtMap * rgbd::eigen::Vector3f(dx, -dx, 3)));
		rgbd::eigen2ptX(frMesh.v(4), rgbd::eigen::Vector3f(curCamPoseWrtMap * rgbd::eigen::Vector3f(dx, dx, 3)));
		frMesh.allocateTriangles(4);
		frMesh.setTriangle(0, triangulatedMesh::triangle{{{0, 2, 1}}});
		frMesh.setTriangle(1, triangulatedMesh::triangle{{{0, 3, 2}}});
		frMesh.setTriangle(2, triangulatedMesh::triangle{{{0, 4, 3}}});
		frMesh.setTriangle(3, triangulatedMesh::triangle{{{0, 1, 4}}});

		OpenRAVE::TriMesh raveMeshMarkers;
		triangulatedMesh2openraveMesh(frMesh, curMapPoseWrtRaveWorld, raveMeshMarkers);
		OpenRAVE::KinBodyPtr raveMeshPtr = OpenRAVE::RaveCreateKinBody(visEnv); //it won't work if I ask for a non-empty name here
		ASSERT_ALWAYS(raveMeshPtr);
		raveMeshPtr->InitFromTrimesh(raveMeshMarkers, true/* visible */);
		setRaveBodyColor(raveMeshPtr, rgbd::eigen::Vector3f(0, 1, 0), .8);
		raveMeshPtr->SetName("frustum"); //if I don't set a name, AddKinBody() will fail

		raveEnvLock envLock(visEnv, "addMarkerMeshToEnv");
		const auto bodyptr = visEnv->GetKinBody("frustum");
		if(bodyptr) visEnv->Remove(bodyptr);
		visEnv->Add(raveMeshPtr); //if I use the default openrave collision checker (ode) for this env, this call takes many seconds, almost all for adding the body to the collision checker
	}

	if(false) //TODO ?
	{
		/*
		 * visualize all views (voxel/orientation combos) we've planned to
		 *
		 * TODO also show overviewed views?
		 */
		triangulatedMesh ovMesh;
		const auto params_volume = onlineMapper->getCurSceneMapParams().volume;
		cout << "visualizing " << allViewsPlannedTo.size() << " views planned to" << endl;
		for(size_t i = 0; i < allViewsPlannedTo.size(); i++)
		{
			//put a marker in this voxel
			triangulatedMesh markerMesh = std::move(makeOctahedronMesh(rgbd::eigen::Vector3f(allViewsPlannedTo[i][0] + .5, allViewsPlannedTo[i][1] + .5, allViewsPlannedTo[i][2] + .5) * params_volume.cell_size, params_volume.cell_size * 2/* size */, rgbd::eigen::Vector3f(0, 0, 0)));
			ovMesh.append(markerMesh);
		}

		OpenRAVE::TriMesh raveMeshMarkers;
		triangulatedMesh2openraveMesh(ovMesh, curMapPoseWrtRaveWorld, raveMeshMarkers);
		OpenRAVE::KinBodyPtr raveMeshPtr = OpenRAVE::RaveCreateKinBody(visEnv); //it won't work if I ask for a non-empty name here
		ASSERT_ALWAYS(raveMeshPtr);
		raveMeshPtr->InitFromTrimesh(raveMeshMarkers, true/* visible */);
		setRaveBodyColor(raveMeshPtr, rgbd::eigen::Vector3f(0, 0, 0), 0);
		raveMeshPtr->SetName("overviewedViewMarkers"); //if I don't set a name, AddKinBody() will fail

		raveEnvLock envLock(visEnv, "addMarkerMeshToEnv");
		const auto bodyptr = visEnv->GetKinBody("overviewedViewMarkers");
		if(bodyptr) visEnv->Remove(bodyptr);
		visEnv->Add(raveMeshPtr); //if I use the default openrave collision checker (ode) for this env, this call takes many seconds, almost all for adding the body to the collision checker
	}

	if(curTarget)
	{
		/*
		 * visualize cur target
		 */
		cout << "visualizing cur target" << endl;

		const auto params_volume = onlineMapper->getCurSceneMapParams().volume;
		const rgbd::eigen::Vector3f tgtPosWrtMap = curTarget->camPoseWrtMap.get() * rgbd::eigen::Vector3f(0, 0, curTarget->targetInfo.viewDist);
		std::array<uint32_t, 3> tgtVoxel;
		for(size_t k = 0; k < 3; k++) tgtVoxel[k] = tgtPosWrtMap[k] / params_volume.cell_size;
		triangulatedMesh ovMesh = std::move(makeOctahedronMesh(rgbd::eigen::Vector3f(tgtVoxel[0] + .5, tgtVoxel[1] + .5, tgtVoxel[2] + .5) * params_volume.cell_size, params_volume.cell_size * 3/* size */, rgbd::eigen::Vector3f(1, 0, 0)));

		OpenRAVE::TriMesh raveMeshMarkers;
		triangulatedMesh2openraveMesh(ovMesh, curMapPoseWrtRaveWorld, raveMeshMarkers);
		OpenRAVE::KinBodyPtr raveMeshPtr = OpenRAVE::RaveCreateKinBody(visEnv); //it won't work if I ask for a non-empty name here
		ASSERT_ALWAYS(raveMeshPtr);
		raveMeshPtr->InitFromTrimesh(raveMeshMarkers, true/* visible */);
		setRaveBodyColor(raveMeshPtr, rgbd::eigen::Vector3f(1, 0, 0), 0);
		raveMeshPtr->SetName("tgtPos"); //if I don't set a name, AddKinBody() will fail

		raveEnvLock envLock(visEnv, "addMarkerMeshToEnv");
		const auto bodyptr = visEnv->GetKinBody("tgtPos");
		if(bodyptr) visEnv->Remove(bodyptr);
		visEnv->Add(raveMeshPtr); //if I use the default openrave collision checker (ode) for this env, this call takes many seconds, almost all for adding the body to the collision checker
	}

	t.stop("visualize planning state in rave");
}

	reportMemUsage("end processMaps");
	t2.stop("process maps before view selection");
}

#ifdef UNUSED
/*
 * inverse kinematics
 *
 * tgtPoseWrtRobotBase will be applied to all poses in the vector ('tgt' is user-defined)
 *
 * return an ik solution for each pose, or empty if no solution was found
 */
std::vector<std::vector<OpenRAVE::dReal>> onlineActiveModeler::runIKForPoses(const rgbd::eigen::Affine3f& tgtPoseWrtRobotBase, const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& camPosesWrtTgt,
	const bool doCollisionChecking)
{
	std::vector<std::vector<OpenRAVE::dReal>> ikSolutions(camPosesWrtTgt.size());

	/*
	 * with openrave's bullet collision checker, even with 14 threads this takes .95s; ODErave even w/ one thread is .65s for the same # of poses
	 */

	rgbd::timer t;
	const size_t numThreads = 1;//raveEnvCopies.size(); //works fine with 1 thread, and is twice as fast as when we use the same env as for display; TODO haven't gotten it to not seg fault when using multiple threads here -- ??
	rgbd::threadGroup tg(numThreads);
	for(size_t m = 0; m < numThreads; m++)
		tg.addTask([&,m]()
			{
	OpenRAVE::RobotBasePtr robot = raveEnvCopies[m]->GetRobot(robotCamHandler->getRobotName());
	OpenRAVE::RobotBase::ManipulatorPtr manip = robot->GetActiveManipulator();
	for(size_t i = m; i < camPosesWrtTgt.size(); i += numThreads)
	{
		/*
		 * run ik
		 *
		 * the actual ik running takes up almost all the time in this loop
		 */

		const rgbd::eigen::Affine3f camPoseWrtRobotBase = tgtPoseWrtRobotBase * camPosesWrtTgt[i];

		//quickly filter out poses too far from the base to be plausibly reachable
		float dx, da;
		xf::transform_difference(rgbd::eigen::Affine3f::Identity(), camPoseWrtRobotBase, dx, da);
		if(dx > 1.1) continue;

		const rgbd::eigen::Affine3f wristPoseWrtRobotBase = camPoseWrtRobotBase * robotCamHandler->getCamPoseWrtAttachmentLink().inverse();
#if 1 //debugging
		if(!wristPoseWrtRobotBase.linear().isUnitary(1e-4) || fabs(wristPoseWrtRobotBase.linear().determinant() - 1) > 1e-4)
		{
			cout << "possibly bad wristPoseWrtRobotBase " << i << ":" << endl << wristPoseWrtRobotBase.linear() << endl;
			cout << "cpwrb " << endl << camPoseWrtRobotBase.matrix() << endl;
			cout << "cpww " << endl << robotCamHandler->getCamPoseWrtAttachmentLink().inverse().matrix() << endl;
			cout << "cpwt " << endl << camPosesWrtTgt[i].matrix() << endl;
			cout << "tpwrb " << endl << tgtPoseWrtRobotBase.matrix() << endl;
			if(!(camPosesWrtTgt[i].linear().isUnitary(1e-4) && fabs(camPosesWrtTgt[i].linear().determinant() - 1) < 1e-4)) exit(1);
		}
#endif
		const OpenRAVE::RaveTransform<OpenRAVE::dReal> ikTgtPoseWrtRaveWorld = robot->GetTransform() * eigenXform2raveXform(wristPoseWrtRobotBase) * manip->GetLocalToolTransform();
		const OpenRAVE::IkParameterization ikTgt(ikTgtPoseWrtRaveWorld, OpenRAVE::IKP_Transform6D);
		vector<OpenRAVE::dReal> soln;
		bool success = false;
	{
	//	OpenRAVE::EnvironmentMutex::scoped_lock lock(raveEnvCopies[m]->GetMutex()); // lock environment; unnec if we use envs we don't also use for anything else
		try
		{
			/*
			 * collision checking is very slow; as of 20131210 I can do much better running similar stuff in cgal myself (about a 6x speedup even without optimizing the cgal-using code; much more after that)
			 */
			success = manip->FindIKSolution(ikTgt, soln, doCollisionChecking ? OpenRAVE::IKFO_CheckEnvCollisions : 0); //doing collision checking here is very slow
		}
		catch(const std::exception& x)
		{
			cout << "exception in ik: " << x.what() << endl;
			cout << "camPoseWrtBase" << endl << camPoseWrtRobotBase.matrix() << endl;
			exit(1);
			ASSERT_ALWAYS(false);
		}
		if(success)
		{
			ikSolutions[i] = soln;
		}
	}

	}
			});
	tg.wait();

	return ikSolutions;
}
#endif

/**************************************************************************************************************************************************************************************************************************************
 * suggesting next poses
 */

/*
 * one pose per triangle, with z pointing along the surface normal and y vaguely aligned with the global up direction
 *
 * return cam poses wrt the mesh, along with validity (false == invalid) for each one
 */
std::tuple<std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>, std::vector<uint8_t>> getMeshTriangleViewTargetPoses(const std::shared_ptr<triangulatedMesh>& mesh, const std::vector<size_t>& triangleIndices,
	const rgbd::eigen::Vector3f& mapUpVec)
{
	std::tuple<std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>, std::vector<uint8_t>> result;
	std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& tgtPosesWrtMesh = std::get<0>(result);
	std::vector<uint8_t>& tgtPoseValidity = std::get<1>(result);
	tgtPosesWrtMesh.resize(triangleIndices.size());
	tgtPoseValidity.resize(triangleIndices.size(), false);

	const std::vector<triangulatedMesh::triangle>& tris = mesh->getTriangles();
	const size_t numThreads = getSuggestedThreadCount(2, 2);
	rgbd::threadGroup tg(numThreads);
	const std::vector<unsigned int> tindices = partitionEvenly(triangleIndices.size(), numThreads);
	for(size_t m = 0; m < numThreads; m++)
		tg.addTask([&,m]()
			{
				for(size_t k = tindices[m]; k < tindices[m + 1]; k++)
				{
					const size_t i = triangleIndices[k];
					/*
					 * create a pose whose translation is the triangle's centroid and whose z axis is the triangle's normal, with y closely aligned with the given vertical
					 */
					const rgbd::eigen::Vector3f x0 = rgbd::ptX2eigen<Vector3f>(mesh->v(tris[i].v[0])), x1 = rgbd::ptX2eigen<Vector3f>(mesh->v(tris[i].v[1])), x2 = rgbd::ptX2eigen<Vector3f>(mesh->v(tris[i].v[2]));
					if((x0 - x1).squaredNorm() > 1e-8 && (x0 - x2).squaredNorm() > 1e-8 && (x1 - x2).squaredNorm() > 1e-8) //we can't estimate transforms reliably from points too close to each other; you end up with nans
					{
						rgbd::eigen::Vector3f zDir = mesh->getTriangleNormal(i);
						if(!std::isnan(zDir.x()) && !std::isnan(zDir.y()) && !std::isnan(zDir.z()))
						{
							const rgbd::eigen::Vector3f centroid = (x0 + x1 + x2) / 3;

							ASSERT_ALWAYS(fabs(zDir.squaredNorm() - 1) < 1e-6);
							//find an up (y) direction for this target that's orthogonal to the z dir
							rgbd::eigen::Vector3f upVec = mapUpVec;
							//perturb it with small rotations until we find one not too close to the z dir
							while(fabs(upVec.dot(zDir)) > .98) upVec = -(rgbd::eigen::AngleAxisf(M_PI * .1, rgbd::eigen::Vector3f((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX)) * mapUpVec).normalized();
							upVec -= upVec.dot(zDir) * zDir;
							upVec.normalize();
							const rgbd::eigen::Vector3f xDir = zDir.cross(upVec);

							rgbd::eigen::Affine3f pose;
							pose.matrix().block<3, 1>(0, 0) = xDir;
							pose.matrix().block<3, 1>(0, 1) = -upVec;
							pose.matrix().block<3, 1>(0, 2) = zDir;
							pose.matrix().block<3, 1>(0, 3) = centroid;
							tgtPosesWrtMesh[k] = pose;
							tgtPoseValidity[k] = true;
						}
					}
				}
			});
	tg.wait();

	return result;
}

/*
 * one pose per triangle
 *
 * return cam poses wrt the mesh, along with validity (false == invalid) for each one
 */
std::tuple<std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>, std::vector<uint8_t>> suggestPosesViewingMeshTriangles(const std::shared_ptr<triangulatedMesh>& mesh, const std::vector<size_t>& triangleIndices)
{
	std::tuple<std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>, std::vector<uint8_t>> result;
	std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& camPosesWrtMesh = std::get<0>(result);
	std::vector<uint8_t>& camPoseValidity = std::get<1>(result);
	camPosesWrtMesh.resize(triangleIndices.size());
	camPoseValidity.resize(triangleIndices.size(), false);

	const float viewDist = .5; //TODO parameterize

	const std::vector<triangulatedMesh::triangle>& tris = mesh->getTriangles();
	const size_t numThreads = getSuggestedThreadCount(2, 2);
	rgbd::threadGroup tg(numThreads);
	const std::vector<unsigned int> tindices = partitionEvenly(triangleIndices.size(), numThreads);
	for(size_t m = 0; m < numThreads; m++)
		tg.addTask([&,m]()
			{
				for(size_t k = tindices[m]; k < tindices[m + 1]; k++)
				{
					const size_t i = triangleIndices[k];
					/*
					 * propose a camera pose looking down the normal at the surface from pretty close
					 *
					 * TODO 20131205 why are they all 90 degrees too far clockwise?
					 */
					const rgbd::eigen::Vector3f x0 = rgbd::ptX2eigen<Vector3f>(mesh->v(tris[i].v[0])), x1 = rgbd::ptX2eigen<Vector3f>(mesh->v(tris[i].v[1])), x2 = rgbd::ptX2eigen<Vector3f>(mesh->v(tris[i].v[2]));
					if((x0 - x1).squaredNorm() > 1e-8 && (x0 - x2).squaredNorm() > 1e-8 && (x1 - x2).squaredNorm() > 1e-8) //we can't estimate transforms reliably from points too close to each other; you end up with nans
					{
						const rgbd::eigen::Vector3f lookAt = (x0 + x1 + x2) / 3;
						rgbd::eigen::Vector3f lookDir = -mesh->getTriangleNormal(i);
						if(!std::isnan(lookDir.x()) && !std::isnan(lookDir.y()) && !std::isnan(lookDir.z()))
						{
							ASSERT_ALWAYS(fabs(lookDir.squaredNorm() - 1) < 1e-6);
							rgbd::eigen::Vector3f upVec = -rgbd::eigen::Vector3f::UnitY();
							if(fabs(upVec.dot(lookDir)) > .95) upVec = -rgbd::eigen::Vector3f(0, 1, .3).normalized();
							upVec -= upVec.dot(lookDir) * lookDir;
							upVec.normalize();
							const rgbd::eigen::Vector3f rightAxis = lookDir.cross(upVec);
							const rgbd::eigen::Vector3f eyePos = lookAt - viewDist * lookDir;
							rgbd::eigen::Affine3f camPose;
							camPose.matrix().block<3, 1>(0, 0) = rightAxis;
							camPose.matrix().block<3, 1>(0, 1) = -upVec;
							camPose.matrix().block<3, 1>(0, 2) = lookDir;
							camPose.matrix().block<3, 1>(0, 3) = eyePos;
							camPosesWrtMesh[k] = camPose;
							camPoseValidity[k] = true;
						}
					}
				}
			});
	tg.wait();

	return result;
}

#ifdef UNUSED
/*
 * return a list of cam poses wrt the mesh
 *
 * pick one frontier triangle and propose many poses looking at it
 */
std::vector<onlineActiveModeler::suggestedCamPoseInfo> onlineActiveModeler::suggestPosesViewingNearestFrontierTriangle(const rgbd::eigen::Affine3f& curMapPoseWrtRaveWorld,
	const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<size_t>& frontierTrianglesVec)
{
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);

	std::vector<onlineActiveModeler::suggestedCamPoseInfo> suggestedPoses;
	if(frontierTrianglesVec.empty())
	{
		cout << "WARNING: no frontier triangles in suggestPosesViewingNearestFrontierTriangle()" << endl;
	}
	else
	{

	/*
	 * get poses looking at all frontier triangles
	 */
	const float viewDist = .4; //TODO ?
	std::vector<size_t> selectedTriangleIndices; //indices of mesh triangles for which we suggest views
{
	std::vector<uint8_t> camPoseValidity;
	std::tie(camPosesWrtMap, camPoseValidity) = std::move(suggestPosesViewingMeshTriangles(curMapMesh, frontierTrianglesVec));

	std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> camPosesWrtMap2;
	copySelected(camPosesWrtMap, camPoseValidity, std::back_inserter(camPosesWrtMap2));
	camPosesWrtMap = std::move(camPosesWrtMap2);
	for(size_t i = 0; i < camPoseValidity.size(); i++)
		if(camPoseValidity[i])
			selectedTriangleIndices.push_back(frontierTrianglesVec[i]);
}

	/*
	 * get configurations corresponding to all these poses
	 */
	const std::vector<std::vector<OpenRAVE::dReal>> viewingConfigurations = getViewingConfigurationsForPoses(curMapPoseWrtRaveWorld, camPosesWrtMap, false/* collision checking */);

	/*
	 * sort poses by distance in configuration space from where we are to ik solution
	 *
	 * none of the wam's joints moves more than one full circle, so no periodicity problems
	 */
	const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();
	std::vector<std::pair<size_t, float>> indicesWithScores(viewingConfigurations.size()); //indices into ikSolutions; score = configuration-space distance
	static const float INVALID_DIST = 1e30; //very large
	for(size_t i = 0; i < viewingConfigurations.size(); i++)
	{
		indicesWithScores[i].first = i;
		if(viewingConfigurations[i].empty())
			indicesWithScores[i].second = INVALID_DIST; //make sure indices with no ik solutions will sort last
		else
		{
			float dist = 0;
			for(size_t j = 0; j < viewingConfigurations[i].size(); j++) dist += sqr(viewingConfigurations[i][j] - curConfig[j]);
			indicesWithScores[i].second = dist;
		}
	}
	std::sort(indicesWithScores.begin(), indicesWithScores.end(), [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b){return a.second < b.second;}); //sort by distance increasing
#if 1 //debugging
		std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWrtMapWithScores(camPosesWrtMap.size());
		for(size_t i = 0; i < camPosesWrtMap.size(); i++)
		{
			camPosesWrtMapWithScores[i].first = camPosesWrtMap[i];
			camPosesWrtMapWithScores[i].second = (indicesWithScores[i].second == INVALID_DIST) ? 1 : (float)i / camPosesWrtMap.size();
		}
		visualizeMeshWithCameraPoses(*curMapMesh, camPosesWrtMapWithScores, framePosesWrtCurMap[0] * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -1))/* viewingPoseWrtMap */, true/* show cur pose */,
			"", outdir / (boost::format("ikScoredPoses%1%.ply") % frameIndex).str());
#endif

	/*
	 * propose a lot of poses looking at the triangle whose canonical pose is closest in configuration space, in the hopes that one or more of them will be reachable in a non-colliding pose
	 */
	if(indicesWithScores[0].second < INVALID_DIST) //if we got an config suggestion for at least one pose
	{
		std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> camPosesWrtMap2;

		const rgbd::eigen::Affine3f camPoseWrtMap = camPosesWrtMap[indicesWithScores[0].first];
		const rgbd::eigen::Vector3f lookAtPt = camPoseWrtMap * rgbd::eigen::Vector3f(0, 0, viewDist), //recover the position and normal in map space of the surface we want to see
			normal = camPoseWrtMap.linear() * rgbd::eigen::Vector3f(0, 0, -1);
		//find an arbitrary dir orthogonal to the normal to be theta=0
		rgbd::eigen::Vector3f refDir = rgbd::eigen::Vector3f::UnitX();
		if(fabs(refDir.dot(normal)) > .95) refDir = rgbd::eigen::Vector3f::UnitY();
		refDir = (refDir - refDir.dot(normal) * normal).normalized();
		const rgbd::eigen::Vector3f refDir2 = normal.cross(refDir);

		for(float newViewDist = .4; newViewDist < 1; newViewDist += .1)
			for(float theta = 0; theta < 2 * M_PI; theta += M_PI / 8) //rotate around normal
				for(float phi = 0; phi < M_PI / 2; phi += M_PI / 8) //0 = along normal, pi = completely oblique
				{
					const rgbd::eigen::Vector3f eyePos = lookAtPt + (rgbd::eigen::AngleAxisf(theta, normal) * rgbd::eigen::AngleAxisf(phi, refDir2) * normal) * newViewDist;
					const rgbd::eigen::Vector3f lookDir = (lookAtPt - eyePos).normalized();
					rgbd::eigen::Vector3f upVec = normal;
					if(fabs(upVec.dot(lookDir)) > .98) upVec = rgbd::eigen::AngleAxisf(.2 * M_PI, refDir2) * normal;
					upVec -= upVec.dot(lookDir) * lookDir;
					upVec.normalize();
					const rgbd::eigen::Vector3f rightAxis = lookDir.cross(upVec);
					for(float psi = 0; psi < 2 * M_PI; psi += M_PI / 2) //rotate the camera about the optical axis
					{
						rgbd::eigen::Affine3f camPose; //wrt robot base
						camPose.matrix().block<3, 1>(0, 0) = rgbd::eigen::AngleAxisf(psi, lookDir) * rightAxis;
						camPose.matrix().block<3, 1>(0, 1) = rgbd::eigen::AngleAxisf(psi, lookDir) * -upVec;
						camPose.matrix().block<3, 1>(0, 2) = lookDir;
						camPose.matrix().block<3, 1>(0, 3) = eyePos;
						camPosesWrtMap2.push_back(camPose);
					}
				}

#if 0 //debugging
		std::shared_ptr<triangulatedMesh> curMeshEdited(new triangulatedMesh(*curMapMesh));
		const size_t triIndex = selectedTriangleIndices[indicesWithScores[0].first];
		const auto& tris = curMeshEdited->getTriangles();
		for(int q = 0; q < 3; q++) curMeshEdited->v(tris[triIndex].v[q]).rgb = rgbd::packRGB(255, 0, 0);
		std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWrtMapWithScores(camPosesWrtMap2.size());
		for(size_t i = 0; i < camPosesWrtMap2.size(); i++)
		{
			camPosesWrtMapWithScores[i].first = camPosesWrtMap2[i];
			camPosesWrtMapWithScores[i].second = 1;
		}
		visualizeMeshWithCameraPoses(curMeshEdited, camPosesWrtMapWithScores, framePosesWrtCurMap[0] * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -1))/* viewingPoseWrtMap */, true/* show cur pose */,
			"", outdir / (boost::format("allPossiblePoses%1%.ply") % frameIndex).str());
#endif

		/*
		 * de-suggest poses that can't see the target triangle
		 */
	{
		/*
		 * color just the target triangle
		 */
		const triangulatedMeshRenderer::triangleColoringFunc getTriangleColor = [&](const uint32_t index, const std::array<rgbd::pt, 3>& pts)
			{
				std::array<uint8_t, 4> rgba = {{255, 0, 0, 255}};
				if(index == selectedTriangleIndices[indicesWithScores[0].first]) rgba[1] = 255;
				else rgba[1] = 0;
				return rgba;
			};
		glContext->lock();
		glContext->makeCurrent();
		std::shared_ptr<triangulatedMeshRenderer> meshRenderer(new triangulatedMeshRenderer(*curMapMesh, getTriangleColor, camParams));
		glContext->unlock();
		sceneRenderer->acquire();
		sceneRenderer->setRenderFunc([&meshRenderer](const rgbd::eigen::Affine3f& camPose){meshRenderer->render(camPose);});
		const vector<float> viewScores = std::move(sceneRenderer->renderAndScore(camPosesWrtMap2));
		sceneRenderer->restoreRenderFunc();
		sceneRenderer->release();
		camPosesWrtMap.clear();
		for(size_t i = 0; i < camPosesWrtMap2.size(); i++)
			if(viewScores[i] > 0)
				camPosesWrtMap.push_back(camPosesWrtMap2[i]);
	}

#if 0 //debugging
	{
		std::shared_ptr<triangulatedMesh> curMeshEdited(new triangulatedMesh(*curMapMesh));
		const size_t triIndex = selectedTriangleIndices[indicesWithScores[0].first];
		const auto& tris = curMeshEdited->getTriangles();
		for(int q = 0; q < 3; q++) curMeshEdited->v(tris[triIndex].v[q]).rgb = rgbd::packRGB(255, 0, 0);
		std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWrtMapWithScores(camPosesWrtMap.size());
		for(size_t i = 0; i < camPosesWrtMap.size(); i++)
		{
			camPosesWrtMapWithScores[i].first = camPosesWrtMap[i];
			camPosesWrtMapWithScores[i].second = 1;
		}
		visualizeMeshWithCameraPoses(curMeshEdited, camPosesWrtMapWithScores, framePosesWrtCurMap[0] * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -1))/* viewingPoseWrtMap */, true/* show cur pose */,
			"", outdir / (boost::format("allPossiblePosesVisible%1%.ply") % frameIndex).str());
	}
#endif

	}

	}
	return suggestedPoses;
}
#endif

#ifdef UNUSED
/*
 * return a list of cam poses wrt the map
 *
 * propose poses near the current one
 */
std::vector<onlineActiveModeler::suggestedCamPoseInfo> onlineActiveModeler::suggestPosesNearCurrent(const rgbd::eigen::Affine3f& curMapPoseWrtRaveWorld)
{
	std::vector<onlineActiveModeler::suggestedCamPoseInfo> suggestedPoses;

	//find poses near the end of our currently planned path
	std::vector<OpenRAVE::dReal> seedConfig;
{
	std::lock_guard<std::mutex> lock(planStartConfigMux);
	if(resetPlanStartConfig) seedConfig = robotCamHandler->getLatestConfiguration();
	else seedConfig = planStartConfig;
}

	const rgbd::eigen::Affine3f curCamPoseWrtRaveWorld = robotCamHandler->getRobotBasePoseWrtRaveWorld(seedConfig) * robotCamHandler->getCamPoseWrtRobotBase(seedConfig),
		curCamPoseWrtMap = curMapPoseWrtRaveWorld.inverse() * curCamPoseWrtRaveWorld;
	const float dx = .02, da = .1; //something pretty small; TODO how small is small enough that the wam won't move at all?
	const std::vector<rgbd::eigen::Affine3f> deltaTs =
	{
		rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(-dx, 0, 0)),
		rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(dx, 0, 0)),
		rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, -dx, 0)),
		rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, dx, 0)),
		rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -dx)),
		rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, dx)),
	},
		deltaAs =
	{
		rgbd::eigen::Affine3f(rgbd::eigen::AngleAxisf(-da, rgbd::eigen::Vector3f::UnitX())),
		rgbd::eigen::Affine3f(rgbd::eigen::AngleAxisf(da, rgbd::eigen::Vector3f::UnitX())),
		rgbd::eigen::Affine3f(rgbd::eigen::AngleAxisf(-da, rgbd::eigen::Vector3f::UnitY())),
		rgbd::eigen::Affine3f(rgbd::eigen::AngleAxisf(da, rgbd::eigen::Vector3f::UnitY())),
		rgbd::eigen::Affine3f(rgbd::eigen::AngleAxisf(-da, rgbd::eigen::Vector3f::UnitZ())),
		rgbd::eigen::Affine3f(rgbd::eigen::AngleAxisf(da, rgbd::eigen::Vector3f::UnitZ())),
	};

#if 0 //to get each new pose, apply one delta once
	for(const rgbd::eigen::Affine3f& x : deltaTs)
		camPosesWrtMap.push_back(x * curCamPoseWrtMap);
	for(const rgbd::eigen::Affine3f& x : deltaAs)
		camPosesWrtMap.push_back(rgbd::eigen::Translation3f(curCamPoseWrtMap.translation()) * x * rgbd::eigen::Translation3f(-curCamPoseWrtMap.translation()) * curCamPoseWrtMap);
#elif 1 //to get each new pose, apply up to n deltas, selected semirandomly; hoping this will lead to less random behavior than only using one delta per new pose
	const size_t numNextPoses = 100;
	std::mt19937 rng((unsigned int)time(nullptr));
	std::uniform_int_distribution<int32_t> dist(-3, 3); //for deciding how many of each delta transform to apply
	const auto gen = [&](){return dist(rng);};
	for(size_t i = 0; i < numNextPoses; i++)
	{
		rgbd::eigen::Affine3f camPose = curCamPoseWrtMap;
		const int32_t numDX = gen(), numDY = gen(), numDZ = gen(), numDAX = gen(), numDAY = gen(), numDAZ = gen();
		camPose = camPose * rgbd::eigen::Translation3f(dx * numDX, 0, 0);
		camPose = camPose * rgbd::eigen::Translation3f(0, dx * numDY, 0);
		camPose = camPose * rgbd::eigen::Translation3f(0, 0, dx * numDZ);
		rgbd::eigen::Affine3f rot(rgbd::eigen::AngleAxisf(da * numDAX, rgbd::eigen::Vector3f::UnitX()));
		camPose = camPose * rot;
		rot = rgbd::eigen::Affine3f(rgbd::eigen::AngleAxisf(da * numDAY, rgbd::eigen::Vector3f::UnitY()));
		camPose = camPose * rot;
		rot = rgbd::eigen::Affine3f(rgbd::eigen::AngleAxisf(da * numDAZ, rgbd::eigen::Vector3f::UnitZ()));
		camPose = camPose * rot;
		suggestedCamPoseInfo poseInfo;
		poseInfo.camPoseWrtMap = camPose;
		suggestedPoses.push_back(poseInfo);
	}
#endif

	return suggestedPoses;
}
#endif

/*
 * return a ranked subset of all mesh triangles that we should consider looking at in order
 *
 * negative triangle values mean it's not useful to look at them
 */
std::vector<onlineActiveModeler::viewTargetInfo> onlineActiveModeler::suggestOrderedViewTargets(const rgbd::eigen::Affine3f& curMapPoseWrtRaveWorld,
	const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<float>& triangleValues)
{
	ASSERT_ALWAYS(triangleValues.size() == curMapMesh->numTriangles());

	std::vector<onlineActiveModeler::viewTargetInfo> suggestedTris;

	struct indexNValue
	{
		bool operator < (const indexNValue& i) const {return value > i.value;}

		size_t index;
		float value;
	};

	rgbd::timer t;

	/*
	 * sort triangles by viewing value
	 */
	size_t numUsefulTriangles = 0;
	for(size_t i = 0; i < curMapMesh->numTriangles(); i++)
		if(triangleValues[i] > 0)
			numUsefulTriangles++;
	std::vector<indexNValue> trisInfo(numUsefulTriangles);
	for(size_t i = 0, j = 0; i < curMapMesh->numTriangles(); i++)
		if(triangleValues[i] > 0)
		{
			trisInfo[j].index = i;
			trisInfo[j].value = triangleValues[i];
			j++;
		}
	cout << "got " << numUsefulTriangles << " tris w/ positive value" << endl;
	std::sort(trisInfo.begin(), trisInfo.end()); //by value decreasing
	std::vector<size_t> sortedTriIndices(trisInfo.size());
	for(size_t i = 0; i < trisInfo.size(); i++) sortedTriIndices[i] = trisInfo[i].index;
	t.stop("sort tris by value");

	t.restart();
	const rgbd::eigen::Vector3f mapUpVec = curMapPoseWrtRaveWorld * rgbd::eigen::Vector3f::UnitZ(); //map dir that's the same as rave-world up
	std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> tgtPosesWrtMap;
	std::vector<uint8_t> tgtPoseValidity;
	std::tie(tgtPosesWrtMap, tgtPoseValidity) = std::move(getMeshTriangleViewTargetPoses(curMapMesh, sortedTriIndices, mapUpVec));
	t.stop("suggest poses for valid tris");

	/*
	 * keep only valid poses in the list
	 */
	t.restart();
{
	const std::vector<size_t> validPoseIndices = boolVectorToIndices(tgtPoseValidity);
	std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> tgtPosesWrtMap2(validPoseIndices.size());
	std::vector<size_t> sortedTriIndices2(validPoseIndices.size());
	copySelectedIndices(tgtPosesWrtMap, validPoseIndices, tgtPosesWrtMap2.begin());
	copySelectedIndices(sortedTriIndices, validPoseIndices, sortedTriIndices2.begin());
	tgtPosesWrtMap = std::move(tgtPosesWrtMap2);
	sortedTriIndices = std::move(sortedTriIndices2);
}
	t.stop("take subsets of poses");

	t.restart();
	/*
	 * reduce the list by rejecting triangle centers too close together as look-at points
	 */
{
	std::vector<uint8_t> toKeep(sortedTriIndices.size(), false);
	struct hashIndex3
	{
		size_t operator () (const std::array<int32_t, 3>& i) const
		{
			size_t h = 0;
			boost::hash_combine(h, i[0]);
			boost::hash_combine(h, i[1]);
			boost::hash_combine(h, i[2]);
			return h;
		}
	};
	const auto& tris = curMapMesh->getTriangles();
	std::unordered_set<std::array<int32_t, 3>, hashIndex3> voxelOccupancy; //only occupied voxels have entries
	const float voxelSize = .03; //TODO ?; should be bigger than cur-map voxel size to be useful; also, the bigger, the faster this step will be
	size_t numKeep = 0;
	for(size_t i = 0; i < sortedTriIndices.size(); i++)
	{
		const size_t t = sortedTriIndices[i];
		const rgbd::eigen::Vector3f x0 = rgbd::ptX2eigen<Vector3f>(curMapMesh->v(tris[t].v[0])), x1 = rgbd::ptX2eigen<Vector3f>(curMapMesh->v(tris[t].v[1])), x2 = rgbd::ptX2eigen<Vector3f>(curMapMesh->v(tris[t].v[2]));
		const rgbd::eigen::Vector3f centroid = (x0 + x1 + x2) / 3;
		const int32_t ix = floor(centroid.x() / voxelSize), iy = floor(centroid.y() / voxelSize), iz = floor(centroid.z() / voxelSize);
		if(voxelOccupancy.find(std::array<int32_t, 3>{{ix, iy, iz}}) == voxelOccupancy.end())
		{
			toKeep[i] = true;
			numKeep++;
			voxelOccupancy.insert(std::array<int32_t, 3>{{ix, iy, iz}});
		}
	}

	suggestedTris.resize(numKeep);
	for(size_t i = 0, j = 0; i < sortedTriIndices.size(); i++)
		if(toKeep[i])
		{
			suggestedTris[j].triIndex = sortedTriIndices[i];
			for(size_t k = 0; k < 3; k++) suggestedTris[j].triVertices[k] = curMapMesh->v(tris[suggestedTris[j].triIndex].v[k]);
			suggestedTris[j].viewDist = .5; //TODO ?
			suggestedTris[j].score = trisInfo[i].value;
			suggestedTris[j].tgtPoseWrtMap = tgtPosesWrtMap[i];
			j++;
		}
}
	t.stop("reject too-close look-at pts");

	return suggestedTris;
}

#ifdef UNUSED
/*
 * return a list of cam poses wrt the map, one to view each triangle given to us
 */
TODO build on suggestOrderedViewTargets
std::vector<onlineActiveModeler::suggestedCamPoseInfo> onlineActiveModeler::suggestPosesViewingValuableTriangles(const rgbd::eigen::Affine3f& curMapPoseWrtRaveWorld,
	const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<float>& triangleValues)
{
	ASSERT_ALWAYS(triangleValues.size() == curMapMesh->numTriangles());

	std::vector<onlineActiveModeler::suggestedCamPoseInfo> suggestedPoses;

	struct indexNValue
	{
		bool operator < (const indexNValue& i) const {return value > i.value;}

		size_t index;
		float value;
	};

	rgbd::timer t;

	/*
	 * sort triangles by viewing value
	 */
	size_t numUsefulTriangles = 0;
	for(size_t i = 0; i < curMapMesh->numTriangles(); i++)
		if(triangleValues[i] > 0)
			numUsefulTriangles++;
	std::vector<indexNValue> trisInfo(numUsefulTriangles);
	for(size_t i = 0, j = 0; i < curMapMesh->numTriangles(); i++)
		if(triangleValues[i] > 0)
		{
			trisInfo[j].index = i;
			trisInfo[j].value = triangleValues[i];
			j++;
		}
	cout << "got " << numUsefulTriangles << " tris w/ positive value" << endl;
	std::sort(trisInfo.begin(), trisInfo.end()); //by value decreasing
	std::vector<size_t> sortedTriIndices(trisInfo.size());
	for(size_t i = 0; i < trisInfo.size(); i++) sortedTriIndices[i] = trisInfo[i].index;
	t.stop("sort tris by value");

	t.restart();
//	const std::vector<size_t> allTriIndices(boost::counting_iterator<size_t>(0), boost::counting_iterator<size_t>(triangleValues.size()); //indices 0..n-1; TODO do the suggestion in a separate thread, then combine w/ pose sort order later?
	std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> camPosesWrtMap;
	std::vector<uint8_t> camPoseValidity;
	std::tie(camPosesWrtMap, camPoseValidity) = std::move(suggestPosesViewingMeshTriangles(curMapMesh, sortedTriIndices));
	t.stop("suggest poses for valid tris");

	/*
	 * keep only valid poses in the list
	 */
	t.restart();
{
	const std::vector<size_t> validPoseIndices = boolVectorToIndices(camPoseValidity);
	std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> camPosesWrtMap2(validPoseIndices.size());
	std::vector<size_t> sortedTriIndices2(validPoseIndices.size());
	copySelectedIndices(camPosesWrtMap, validPoseIndices, camPosesWrtMap2.begin());
	copySelectedIndices(sortedTriIndices, validPoseIndices, sortedTriIndices2.begin());
	camPosesWrtMap = std::move(camPosesWrtMap2);
	sortedTriIndices = std::move(sortedTriIndices2);
}
	t.stop("take subsets of poses");

	t.restart();
	/*
	 * reduce the list by rejecting triangle centers too close together as look-at points
	 */
{
	std::vector<uint8_t> toKeep(camPosesWrtMap.size(), false);
	struct hashIndex3
	{
		size_t operator () (const std::array<int32_t, 3>& i) const
		{
			size_t h = 0;
			boost::hash_combine(h, i[0]);
			boost::hash_combine(h, i[1]);
			boost::hash_combine(h, i[2]);
			return h;
		}
	};
	const auto& tris = curMapMesh->getTriangles();
	std::unordered_set<std::array<int32_t, 3>, hashIndex3> voxelOccupancy; //only occupied voxels have entries
	const float voxelSize = .03; //TODO ?; should be bigger than cur-map voxel size to be useful; also, the bigger, the faster this step will be
	size_t numKeep = 0;
	for(size_t i = 0; i < sortedTriIndices.size(); i++)
	{
		const size_t t = sortedTriIndices[i];
		const rgbd::eigen::Vector3f x0 = rgbd::ptX2eigen<Vector3f>(curMapMesh->v(tris[t].v[0])), x1 = rgbd::ptX2eigen<Vector3f>(curMapMesh->v(tris[t].v[1])), x2 = rgbd::ptX2eigen<Vector3f>(curMapMesh->v(tris[t].v[2]));
		const rgbd::eigen::Vector3f centroid = (x0 + x1 + x2) / 3;
		const int32_t ix = floor(centroid.x() / voxelSize), iy = floor(centroid.y() / voxelSize), iz = floor(centroid.z() / voxelSize);
		if(voxelOccupancy.find(std::array<int32_t, 3>{{ix, iy, iz}}) == voxelOccupancy.end())
		{
			toKeep[i] = true;
			numKeep++;
			voxelOccupancy.insert(std::array<int32_t, 3>{{ix, iy, iz}});
		}
	}

	suggestedPoses.resize(numKeep);
	for(size_t i = 0, j = 0; i < sortedTriIndices.size(); i++)
		if(toKeep[i])
		{
			suggestedPoses[j].camPoseWrtMap = camPosesWrtMap[i];
			suggestedPoses[j].triIndex = sortedTriIndices[i];
			suggestedPoses[j].viewDist = .4; //TODO ensure this is synced with the actual value
			for(size_t k = 0; k < 3; k++) suggestedPoses[j].triVertices[k] = curMapMesh->v(tris[suggestedPoses[j].triIndex].v[k]);
			j++;
		}
}
	t.stop("reject too-close look-at pts");

	return suggestedPoses;
}
#endif

/**************************************************************************************************************************************************************************************************************************************/

/*
 * list triangles at seen/unseen borders in the mesh
 */
std::vector<size_t> onlineActiveModeler::findSurfaceTrianglesBorderingUnseen(const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<uint8_t>& triangleIsSurface) const
{
	std::vector<size_t> surfaceTrianglesBorderingUnseen;
	rgbd::timer t;
	const auto& tris = curMapMesh->getTriangles();
	std::vector<bool> vertexBordersSeen(curMapMesh->numVertices(), false), vertexBordersUnseen(curMapMesh->numVertices(), false);
	for(size_t i = 0; i < tris.size(); i++)
		for(size_t k = 0; k < 3; k++)
		{
			if(!triangleIsSurface[i]) vertexBordersUnseen[tris[i].v[k]] = true;
			else vertexBordersSeen[tris[i].v[k]] = true;
		}
	for(size_t i = 0; i < tris.size(); i++)
		if(triangleIsSurface[i])
			for(size_t k = 0; k < 3; k++)
				if(vertexBordersUnseen[tris[i].v[k]])
				{
					surfaceTrianglesBorderingUnseen.push_back(i);
					break;
				}
	t.stop("compute surfaceTrianglesBorderingUnseen");
	return surfaceTrianglesBorderingUnseen;
}

/*
 * info for computing geodesic distances: return for each triangle the avg distance from the centroid to a vertex
 */
std::vector<float> onlineActiveModeler::getMeshTriangleSizes(const triangulatedMesh& mesh) const
{
	const auto& tris = mesh.getTriangles();
	std::vector<float> triSize(mesh.numTriangles()); //avg dist from centroid to a vertex, per triangle
	for(size_t i = 0; i < mesh.numTriangles(); i++)
	{
		const rgbd::eigen::Vector3f v0 = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(mesh.v(tris[i].v[0])), v1 = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(mesh.v(tris[i].v[1])), v2 = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(mesh.v(tris[i].v[2])),
			centroid = (v0 + v1 + v2) / 3;
		triSize[i] = ((centroid - v0).norm() + (centroid - v1).norm() + (centroid - v2).norm()) / 3;
	}
	return triSize;
}

/*
 * info for computing geodesic distances: return for each triangle the indices of all neighboring triangles, unsorted
 */
std::vector<std::vector<size_t>> onlineActiveModeler::getTriangleNeighbors(const triangulatedMesh& mesh) const
{
	const auto& tris = mesh.getTriangles();
	std::vector<std::vector<size_t>> triNbrs; //triangle index -> indices of nbring triangles, unsorted
	rgbd::timer t;
	std::vector<std::vector<size_t>> trisByVertex(mesh.numVertices());
	for(size_t i = 0; i < tris.size(); i++)
		for(size_t k = 0; k < 3; k++)
			trisByVertex[tris[i].v[k]].push_back(i);
	triNbrs.resize(mesh.numTriangles());
	for(size_t i = 0; i < tris.size(); i++)
	{
		std::unordered_set<size_t> nbrs;
		for(size_t k = 0; k < 3; k++) nbrs.insert(trisByVertex[tris[i].v[k]].begin(), trisByVertex[tris[i].v[k]].end());
		nbrs.erase(i);
		triNbrs[i].insert(triNbrs[i].end(), nbrs.begin(), nbrs.end());
	}
	t.stop("compute triangle nbrs");
	return triNbrs;
}

/*
 * return: pose distance from the given pose wrt the cur cam pose to, for each mesh triangle, a cam pose viewing that triangle
 *
 * daWeight: weight for angular dist vs. position dist used in pose dist; I usually use around .5
 */
std::vector<float> onlineActiveModeler::getPoseDistToCamPosesViewingMeshTriangles(const rgbd::eigen::Affine3f& poseWrtCamToConsider, const float daWeight)
{
	std::vector<float> triPoseDistFromCam(curMapMesh->numTriangles());

	const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();
	const rgbd::eigen::Affine3f curCamPoseWrtMap = curMapPoseWrtRaveWorld.inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(curConfig) * robotCamHandler->getCamPoseWrtRobotBase(curConfig);
	/*
	 * dist between cur cam pose and a cam pose looking directly at the triangle from a known distance
	 */
	const auto& tris = curMapMesh->getTriangles();
	const size_t numThreads = 4; //TODO ?
	rgbd::threadGroup tg(numThreads);
	for(size_t k = 0; k < numThreads; k++)
		tg.addTask([&,k]()
			{
				for(size_t i = k; i < curMapMesh->numTriangles(); i += numThreads)
				{
					const rgbd::eigen::Vector3f triCentroid = (rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[i].v[0])) + rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[i].v[1])) + rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[i].v[2]))) / 3;
					const rgbd::eigen::Vector3f triNormal = curMapMesh->getTriangleNormal(i);
					if(std::isnan(triNormal[0]) || std::isinf(triNormal[0]) || fabs(triNormal.squaredNorm() - 1) > 1e-4) //if invalid normal
						triPoseDistFromCam[i] = 1e6; //TODO really want infinity
					else
					{
						const rgbd::eigen::Vector3f lookatPt = triCentroid;
						const rgbd::eigen::Vector3f viewDir = -triNormal;
						//find the closest upVec on the allowable plane to the upVec of the cur cam pose
						const rgbd::eigen::Vector3f curUpVec = curCamPoseWrtMap.linear() * -rgbd::eigen::Vector3f::UnitY();
						rgbd::eigen::Vector3f upVec = curUpVec - curUpVec.dot(viewDir) * viewDir; //project onto the plane with normal viewDir
						upVec.normalize();
						const rgbd::eigen::Vector3f rightVec = viewDir.cross(upVec);
						const float viewDist = .4; //TODO ?
						const rgbd::eigen::Vector3f eyePos = lookatPt - viewDist * viewDir;
						rgbd::eigen::Affine3f camPoseWrtMap;
						camPoseWrtMap.matrix().block<3, 1>(0, 0) = rightVec;
						camPoseWrtMap.matrix().block<3, 1>(0, 1) = -upVec;
						camPoseWrtMap.matrix().block<3, 1>(0, 2) = viewDir;
						camPoseWrtMap.matrix().block<3, 1>(0, 3) = eyePos;

						float dx, da;
						xf::transform_difference(curCamPoseWrtMap * poseWrtCamToConsider, camPoseWrtMap, dx, da);
						triPoseDistFromCam[i] = dx + daWeight * da;
					}
				}
			});
	tg.wait();

	return triPoseDistFromCam;
}

/*
 * return an approximate distance, in meters, of each triangle along mesh surfaces to the nearest high-p(m) triangle
 *
 * maxDist: only find triangles up to this far (in m) from moved surfaces; return some very large distance for other triangles
 */
std::vector<float> onlineActiveModeler::getUnseenTrianglesGeodesicDistToMovedSurface(const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<float>& triSizes, const std::vector<std::vector<size_t>>& triNbrs,
	const sumLogprobsCombined& meshDiffingSums, const float maxDist) const
{
	ASSERT_ALWAYS(meshDiffingSums.size() == curMapMesh->numTriangles());
	ASSERT_ALWAYS(onlineMapper->getCurSceneMapParams().volume_modeler.model_type == MODEL_SINGLE_VOLUME); //grids give you disconnected meshes, so geodesic distance will be screwed up
	rgbd::timer t;
	const rgbd::eigen::VectorXd meshMovedProbs = meshDiffingSums.movedProbs();
	std::vector<float> triGeodesicDistToMovedSurface(curMapMesh->numTriangles(), 1e5); //very large distance = not set yet
	/*
	 * run BFS over triangles
	 */
{
	std::array<std::vector<float>, 2> triGeodesicDists; //to be read on alternate iterations: read buffer # iter%2
	std::array<std::vector<uint8_t>, 2> activeTris; //which tris might need updating on each iteration (0 or 1), to be read on alternate iterations: read buffer # iter%2
	for(size_t i = 0; i < 2; i++)
	{
		triGeodesicDists[i].resize(curMapMesh->numTriangles(), 1e5); //very large distance = not set yet
		activeTris[i].resize(curMapMesh->numTriangles(), 0);
	}
	//set distances to moved to 0 for tris marked moved
	for(size_t i = 0; i < curMapMesh->numTriangles(); i++)
		if(meshMovedProbs[i] > .6/* TODO ? */)
		{
			triGeodesicDists[0][i] = 0;
			for(size_t n : triNbrs[i])
				if(!triangleIsSurface[n]) //if unseen
					activeTris[0][n] = 1;
		}
	size_t numUpdatesOnLastIter;
	size_t iter = 0;
	do
	{
		numUpdatesOnLastIter = 0;
		const int32_t ii = iter % 2; //which buffer set we're reading this iter
		std::fill(activeTris[1 - ii].begin(), activeTris[1 - ii].end(), 0);
		triGeodesicDists[1 - ii] = triGeodesicDists[ii];
		int numActive = 0;
		for(size_t i = 0; i < curMapMesh->numTriangles(); i++)
			if(activeTris[ii][i])
			{
				bool updated = false;
				for(size_t n : triNbrs[i])
				{
					const float possibleNewDist = triGeodesicDists[ii][n] + triSizes[n] + triSizes[i];
					if(possibleNewDist < triGeodesicDists[ii][i]
						&& possibleNewDist < maxDist) //don't update if we'd go over the max distance we want to measure
					{
						triGeodesicDists[1 - ii][i] = possibleNewDist;
						updated = true;
					}
				}
				if(updated)
				{
					numUpdatesOnLastIter++;
					for(size_t n : triNbrs[i])
						if(!triangleIsSurface[n]) //if unseen
							activeTris[1 - ii][n] = 1;
				}
				numActive++;
			}
		cout << iter << ": " << numActive << " active tris; " << numUpdatesOnLastIter << " updates" << endl;
		iter++;
	}
	while(numUpdatesOnLastIter > 0);
	cout << "ran " << iter << " iters in finding geodesic dists on mesh" << endl;
	triGeodesicDistToMovedSurface = std::move(triGeodesicDists[iter % 2]);
}
	t.stop("get distances from moved surfaces");
	return triGeodesicDistToMovedSurface;
}

/*
 * return: approximate traversal cost, in meter-equivalents, from cur cam pose to each provided pose
 */
std::vector<float> onlineActiveModeler::getTraversalCostsToCamPoses(const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<OpenRAVE::dReal>& curConfig,
	const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& camPosesWrtMap)
{
	const float hugeTraversalCost = 1e10; //very large; TODO ?
	std::vector<float> traversalCosts(camPosesWrtMap.size());

	/*
	 * define a graph with V + P + 1 vertices (in that order): one for each of V free voxels in the map, one for each of P poses viewing mesh triangles, and one for the cur cam pose;
	 * calculate shortest paths in that graph from the cur pose to each other cam pose
	 */

	std::vector<rgbd::eigen::Vector3f> vertices; //cam positions
	std::vector<rgbd::eigen::Affine3f> vertexPoses; //cam poses

	const rgbd::eigen::Affine3f curCamPoseWrtRaveWorld = robotCamHandler->getRobotBasePoseWrtRaveWorld(curConfig) * robotCamHandler->getCamPoseWrtRobotBase(curConfig);
	const rgbd::eigen::Affine3f curCamPoseWrtMap = curMapPoseWrtRaveWorld.inverse() * curCamPoseWrtRaveWorld;

	rgbd::timer t;
	//get bbox for cur-scene mesh (we won't place graph nodes outside that)
	rgbd::eigen::Vector3f mins = rgbd::eigen::Vector3f::Constant(FLT_MAX), maxes = rgbd::eigen::Vector3f::Constant(-FLT_MAX);
	for(size_t i = 0; i < curMapMesh->numVertices(); i++)
	{
		const rgbd::pt pt = curMapMesh->v(i);
		if(pt.x < mins[0]) mins[0] = pt.x;
		if(pt.y < mins[1]) mins[1] = pt.y;
		if(pt.z < mins[2]) mins[2] = pt.z;
		if(pt.x > maxes[0]) maxes[0] = pt.x;
		if(pt.y > maxes[1]) maxes[1] = pt.y;
		if(pt.z > maxes[2]) maxes[2] = pt.z;
	}
	for(size_t k = 0; k < 3; k++) ASSERT_ALWAYS(maxes[k] >= mins[k]);

	//add voxel-center locations that are free in the map
//	const auto& curSceneMapParams = onlineMapper->getCurSceneMapParams();
	const float sampleVoxelSize = .03;//curSceneMapParams.volume.cell_size; //TODO ?
	const size_t numVoxelsX = (size_t)ceil((maxes[0] - mins[0]) / sampleVoxelSize), numVoxelsY = (size_t)ceil((maxes[1] - mins[1]) / sampleVoxelSize), numVoxelsZ = (size_t)ceil((maxes[2] - mins[2]) / sampleVoxelSize);
	for(size_t iz = 0; iz < numVoxelsZ; iz++)
	{
		const float z = mins[2] + iz * sampleVoxelSize;
		for(size_t iy = 0; iy < numVoxelsY; iy++)
		{
			const float y = mins[1] + iy * sampleVoxelSize;
			for(size_t ix = 0; ix < numVoxelsX; ix++)
			{
				const float x = mins[0] + ix * sampleVoxelSize;
				const rgbd::eigen::Vector3f posWrtMap = rgbd::eigen::Vector3f(x, y, z);
				const rgbd::eigen::Vector3f posWrtRaveWorld = curMapPoseWrtRaveWorld * posWrtMap;
				if(tsdfFreeAtPos(posWrtRaveWorld))
				{
					vertices.push_back(posWrtMap);
					vertexPoses.push_back(rgbd::eigen::Translation3f(posWrtMap) * curCamPoseWrtMap.rotation());
				}
			}
		}
	}
	//add suggested cam poses that are free in the map
	const size_t camPosesVerticesStartIndex = vertices.size();
	for(size_t i = 0; i < camPosesWrtMap.size(); i++)
	{
		const rgbd::eigen::Vector3f camPosWrtMap = camPosesWrtMap[i] * rgbd::eigen::Vector3f::Zero(),
			camPosWrtRaveWorld = curMapPoseWrtRaveWorld * camPosWrtMap;
		if(tsdfFreeAtPos(camPosWrtRaveWorld))
		{
			vertices.push_back(camPosWrtMap);
			vertexPoses.push_back(camPosesWrtMap[i]);
		}
	}
	//add cur cam pose wrt map
	vertices.push_back(curCamPoseWrtMap * rgbd::eigen::Vector3f::Zero());
	vertexPoses.push_back(curCamPoseWrtMap);

	t.stop("create vertex list");

	t.restart();
	//use a kdtree to get nbrs and their distances
	typedef std::pair<size_t, size_t> edgeT;
	std::vector<edgeT> edges;
	std::vector<int> edgeWeights; //distance in mm-equivalents, rounded to nearest int (dijkstra requires integers)
	const boost::shared_ptr<kdtree2> verticesKDTree = rgbd::createKDTree2(vertices);
	const kdtreeNbrhoodSpec nspec = kdtreeNbrhoodSpec::byRadius(1.02/* close to 1 */ * sampleVoxelSize);
	for(size_t i = 0; i < vertices.size(); i++)
	{
		const std::vector<float> qpt = {vertices[i].x(), vertices[i].y(), vertices[i].z()};
		const std::vector<kdtree2_result> nbrs = std::move(searchKDTree(*verticesKDTree, nspec, qpt));
		for(const auto n : nbrs)
		{
			edges.push_back(std::make_pair(i, (size_t)n.idx));
			float dx, da;
			const float daWeight = .5; //TODO ?
			xf::transform_difference(vertexPoses[i], vertexPoses[n.idx], dx, da);
			edgeWeights.push_back((int)rint(1000 * (dx + daWeight * da)));
		}
	}
	t.stop("find graph nbrs");

//	cout << vertices.size() << " vertices, " << edges.size() << " edges" << endl;

	t.restart();
	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, boost::property<boost::edge_weight_t, int >> graphT;
	typedef boost::graph_traits<graphT>::vertex_descriptor vertex_descriptor;
	graphT graph(edges.begin(), edges.end(), edgeWeights.begin(), edges.size());
//	boost::property_map<graphT, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, graph);
	std::vector<int> distances(boost::num_vertices(graph));
	vertex_descriptor s = boost::vertex(vertices.size() - 1, graph);
	boost::dijkstra_shortest_paths(graph, s, boost::distance_map(&distances[0]));
	t.stop("get shortest paths");

	for(size_t i = 0, j = 0; i < camPosesWrtMap.size(); i++)
	{
		const rgbd::eigen::Vector3f camPosWrtMap = camPosesWrtMap[i] * rgbd::eigen::Vector3f::Zero(),
			camPosWrtRaveWorld = curMapPoseWrtRaveWorld * camPosWrtMap;
		if(tsdfFreeAtPos(camPosWrtRaveWorld))
		{
			traversalCosts[i] = distances[camPosesVerticesStartIndex + j] * 1e-3;
			j++;
		}
		else
		{
			traversalCosts[i] = hugeTraversalCost;
		}
	}

	return traversalCosts;
}

/*
 * get differencing info per mesh triangle in the current map
 */
sumLogprobsCombined onlineActiveModeler::getAggregatedDiffingResultsPerCurMeshTriangle(const std::shared_ptr<triangulatedMesh>& curMapMesh)
{
	ASSERT_ALWAYS(shouldKeepDiffingInfoForCurScene()); //if this fails, shouldKeepDiffingInfoForCurScene probably needs updating

	sumLogprobsCombined meshDiffingSums; //indexed by mesh triangle
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	const VolumeModelerAllParams& curSceneParams = onlineMapper->getCurSceneMapParams();
	const float curMapVoxelSize = curSceneParams.volume.cell_size;
	const auto& tris = curMapMesh->getTriangles();

	/*
	 * map cur-scene diffing info from voxels onto triangles of the current mesh
	 */
	meshDiffingSums.init(curMapMesh->numTriangles());
	rgbd::timer t;
	for(size_t i = 0; i < curMapMesh->numTriangles(); i++)
	{
		const rgbd::eigen::Vector3f v0 = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[i].v[0])), v1 = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[i].v[1])), v2 = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[i].v[2])),
			centroid = (v0 + v1 + v2) / 3;
		const int64_t ix = clamp((int)floor(centroid[0] / curMapVoxelSize), 0, curSceneParams.volume.cell_count[0] - 1), //TODO test out-of-boundsness here? once in a while it actually is slightly off (by well under a voxel) -- want to clamp in that case
			iy = clamp((int)floor(centroid[1] / curMapVoxelSize), 0, curSceneParams.volume.cell_count[1] - 1),
			iz = clamp((int)floor(centroid[2] / curMapVoxelSize), 0, curSceneParams.volume.cell_count[2] - 1);
		const auto& voxelInfo = diffingResultWrtCurMap(std::array<int64_t, 3>{{ix, iy, iz}});
		meshDiffingSums.logprobsGivenMoved[i] = voxelInfo.evidenceMoved;
		meshDiffingSums.logprobsGivenNotMoved[i] = voxelInfo.evidenceNotMoved;
	}
	t.stop("map diffing info to mesh tris");

#if 1
	/*
	 * visualize mesh moved probs, for debugging alignment
	 */
	t.restart();
{
	triangulatedMesh visMesh;
	visMesh.allocateVertices(3 * curMapMesh->numTriangles());
	visMesh.allocateTriangles(curMapMesh->numTriangles());
	const auto& triangles = curMapMesh->getTriangles();
	const rgbd::eigen::VectorXd meshMovedProbs = meshDiffingSums.movedProbs();
	for(size_t i = 0; i < curMapMesh->numTriangles(); i++)
	{
		triangulatedMesh::triangle t;
		for(size_t k = 0; k < 3; k++)
		{
			t.v[k] = i * 3 + k;

			rgbd::pt v = curMapMesh->v(triangles[i].v[k]);
			v.rgb = rgbd::packRGB(255, 255 * meshMovedProbs[i], 0);
			visMesh.setVertex(i * 3 + k, v);
		}
		visMesh.setTriangle(i, t);
	}
//	visMesh.writePLY(outdir / (boost::format("meshMovedProbs%1$08d.ply") % frameIndex).str());

	/*
	 * render the mesh from the current camera pose
	 */
	cv::Mat_<cv::Vec3b> bgrImg(camParams.yRes, camParams.xRes);
	glContext->acquire();
{
	const triangulatedMeshRenderer::vertexColoringFunc coloringFunc = getMeshVertexColorFromPointColor;
	std::shared_ptr<triangulatedMeshRenderer> meshRenderer(new triangulatedMeshRenderer(visMesh, coloringFunc, camParams));
	sceneRenderer->acquire();
	sceneRenderer->setRenderFunc([&meshRenderer](const rgbd::eigen::Affine3f& camPose) {meshRenderer->render(camPose);});
	sceneRenderer->render(framePosesWrtCurMap[frameIndex], bgrImg);
	sceneRenderer->restoreRenderFunc();
	sceneRenderer->release();
} //ensure the triangulatedMeshRenderer releases its opengl resources while its context is active
	glContext->release();
	meshValuesImg = bgrImg; //TODO actually put the mesh values into the img, rather than moved probs; for now this is all we need it for, though
}
	t.stop("visualize mesh moved probs");
#endif

	return meshDiffingSums;
}

/*
 * render the mesh from all proposed poses and score each view
 *
 * score function is parameterized:
 * - unseen triangles get a score of unseenValue
 * - border triangles (next to unseen) get a score of borderValue
 * - for frontier triangles, p(m) is multiplied by movednessValue and added to the score
 *
 * return: a score for each cam pose wrt map
 */
std::vector<float> onlineActiveModeler::scorePosesByRendering(const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<uint8_t>& triangleIsSurface, const std::unordered_set<size_t>& borderTriangles,
	const sumLogprobsCombined& meshDiffingSums, const uint8_t unseenValue, const uint8_t borderValue, const uint8_t movednessValue, const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& camPosesWrtMap)
{
	std::vector<float> poseScores;
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	rgbd::timer t;
#if 0
	/*
	 * color by physical color, for debugging
	 */
	const triangulatedMeshRenderer::vertexColoringFunc getTriangleColor = getMeshVertexColorFromPointColor;
#elif 1
	/*
	 * combine various things into the score function
	 */
	rgbd::eigen::VectorXd movedProbs; //per triangle; only valid if using movedness in scoring
	if(movednessValue > 0) movedProbs = meshDiffingSums.movedProbs();
	const triangulatedMeshRenderer::triangleColoringFunc getTriangleColor = [&](const uint32_t index, const std::array<rgbd::pt, 3>& pts)
		{
			std::array<uint8_t, 4> rgba = {{255, 0, 0, 255}};
			if(!triangleIsSurface[index]) //if this triangle isn't a surface
			{
				rgba[1] = unseenValue;

				if(movednessValue > 0) rgba[1] += movednessValue * movedProbs[index];
			}
			else
			{
				//if a border triangle (seen but with unseen nbrs)
				if(borderTriangles.find(index) != borderTriangles.end())
				{
					rgba[1] = borderValue;
				}
			}
			return rgba;
		};
#endif
	glContext->acquire();
{
	std::shared_ptr<triangulatedMeshRenderer> meshRenderer(new triangulatedMeshRenderer(*curMapMesh, getTriangleColor, camParams));
	t.stop("init renderer");
	sceneRenderer->acquire();
	sceneRenderer->setRenderFunc([&meshRenderer](const rgbd::eigen::Affine3f& camPose){meshRenderer->render(camPose);});
	poseScores = std::move(sceneRenderer->renderAndScore(camPosesWrtMap));
	sceneRenderer->restoreRenderFunc();
	sceneRenderer->release();
} //ensure the triangulatedMeshRenderer releases its opengl resources while its context is active
	glContext->release();

	return poseScores;
}

std::shared_ptr<precomputedEnvCollisionData> onlineActiveModeler::precomputeEnvironmentGeometryForCollisionChecking()
{
	std::vector<std::shared_ptr<triangulatedMesh>> meshesForCollisionChecking(staticRaveEnvMeshes.size() + 1);
	std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> envMeshPosesWrtRaveWorld(staticRaveEnvMeshes.size() + 1);
	for(size_t i = 0; i < staticRaveEnvMeshes.size(); i++)
	{
		meshesForCollisionChecking[i] = staticRaveEnvMeshes[i];
		envMeshPosesWrtRaveWorld[i] = rgbd::eigen::Affine3f::Identity();
	}
	meshesForCollisionChecking.back() = curMapMesh;
	envMeshPosesWrtRaveWorld.back() = curMapPoseWrtRaveWorld;
	const std::shared_ptr<precomputedEnvCollisionData> collisionCheckingEnvData = precomputeForCollisionChecking(meshesForCollisionChecking, envMeshPosesWrtRaveWorld);
	return collisionCheckingEnvData;
}

struct onlineActiveModeler::camPoseSuggestionData
{
	std::vector<onlineActiveModeler::viewTargetInfo> suggestedTargets;
	std::shared_ptr<precomputedEnvCollisionData> collisionCheckingEnvData; //computed once at the beginning of each view selection process
	size_t numTargetsProcessedSoFar;
	size_t numPosesProcessedSoFar;
	size_t numConfigsProcessedSoFar;
	size_t numReachablePosesSoFar;
	size_t numNoncollidingConfigsSoFar;
};

/*
 * return: suggested next camera poses, most useful first (might be an empty list), with some sort of scores;
 * return only a few at a time so they don't all have to be tested for reachability if one of the first few will be reachable, which is usually the case
 *
 * if geometricEnvData isn't empty, we'll copy it into the result structure
 *
 * post: each pose suggestion has triIndex set but doesn't have a pose yet
 */
std::shared_ptr<onlineActiveModeler::camPoseSuggestionData> onlineActiveModeler::initCamPoseSuggestion(const std::shared_ptr<precomputedEnvCollisionData>& geometricEnvData)
{
	cout << "starting initCamPoseSuggestion" << endl;
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	rgbd::timer t, t2;

	std::shared_ptr<onlineActiveModeler::camPoseSuggestionData> data(new camPoseSuggestionData);

	/********************************************************************************************************************************************************
	 * set the value of viewing each triangle
	 */

	t.restart();
	std::vector<float> meshTriangleValues(curMapMesh->numTriangles(), 0); //in [0, 1]; values == 0 mean triangles aren't useful to look at

	switch(triValueFunc)
	{
		case triValueFuncType::RANDOM_FRONTIER:
		{
			std::mt19937 rng((uint32_t)time(NULL));
			std::uniform_real_distribution<float> dist(0, 1);

			/*
			 * set value of each triangle
			 */
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i]) //if unseen
				{
					meshTriangleValues[i] = dist(rng);
				}

			break;
		}
		case triValueFuncType::DISTANCE_TO_CAM:
		{
			const rgbd::eigen::Affine3f poseWrtCamToConsider(rgbd::eigen::Translation3f(0, 0, .4/* view dist; TODO ? */)); //the location wrt the camera at which we want targets to be
			const std::vector<float> poseDistancesFromCurPose = getPoseDistToCamPosesViewingMeshTriangles(poseWrtCamToConsider);

#ifdef RUN_DIFFING_SCORING_EXPERIMENT //print # unseen tris for 20140415 diffing-scoring experiment; TODO remove
			const sumLogprobsCombined meshDiffingSums = getAggregatedDiffingResultsPerCurMeshTriangle(curMapMesh); //diffing info projected onto each cur-mesh triangles
			const std::vector<float> triSizes = getMeshTriangleSizes(*curMapMesh); //avg dist from centroid to a vertex, per triangle
			const std::vector<std::vector<size_t>> triNbrs = getTriangleNeighbors(*curMapMesh); //triangle index -> indices of nbring triangles, unsorted
			/*
			 * get distance from each tri to nearest high-p(m) tri using bfs on triangles (up to a max distance, for speed)
			 */
			const float maxDist = .03; //only find triangles up to this far (in m) from moved surfaces; TODO ?
			const std::vector<float> triGeodesicDistToMovedSurface = getUnseenTrianglesGeodesicDistToMovedSurface(curMapMesh, triSizes, triNbrs, meshDiffingSums, maxDist);

			/*
			 * print stats about unseen triangles near high-p(m) triangles, for evaluating algorithms for completing objects segmented by change detection
			 */
			size_t numUnseenTrisNearMovedSurface = 0;
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i]) //if unseen
					if(triGeodesicDistToMovedSurface[i] < .04/* TODO ? */)
						numUnseenTrisNearMovedSurface++;
			cout << numUnseenTrisNearMovedSurface << " unseen tris near moved surfaces remain" << endl;
#endif

			/*
			 * set value of each triangle
			 */
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i]) //if unseen
				{
					meshTriangleValues[i] = exp(-1/* TODO ? */ * poseDistancesFromCurPose[i]);
				}

			break;
		}
		case triValueFuncType::DISTANCE_TO_CUR_TARGET:
		{
			/*
			 * look at the nearest frontier to whatever we currently see in the middle of the image
			 */

			const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();
			const rgbd::eigen::Affine3f curCamPoseWrtMap = curMapPoseWrtRaveWorld.inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(curConfig) * robotCamHandler->getCamPoseWrtRobotBase(curConfig);

			/*
			 * render the mesh from the current camera pose
			 */
			boost::multi_array<uint32_t, 2> sampleIDs(boost::extents[camParams.yRes][camParams.xRes]);
			boost::multi_array<float, 2> sampleDepths(boost::extents[camParams.yRes][camParams.xRes]);
			glContext->acquire();
		{
			const triangulatedMeshRenderer::triangleColoringFunc triColoringFunc = getTriangleColorFromID;
			std::shared_ptr<triangulatedMeshRenderer> meshRenderer(new triangulatedMeshRenderer(*curMapMesh, triColoringFunc, camParams));
			sceneRenderer->acquire();
			sceneRenderer->setRenderFunc([&meshRenderer](const rgbd::eigen::Affine3f& camPose) {meshRenderer->render(camPose);});
			projectSceneSamplesIntoCamera(*sceneRenderer, camParams, curCamPoseWrtMap, sampleIDs, sampleDepths);
			sceneRenderer->restoreRenderFunc();
			sceneRenderer->release();
		} //ensure the triangulatedMeshRenderer releases its opengl resources while its context is active
			glContext->release();

			rgbd::eigen::Affine3f poseWrtCamToConsider;
			bool poseValid = false;
			if(sampleIDs[camParams.yRes / 2][camParams.xRes / 2] > 0) //if there's a triangle at the center pixel
			{
				const std::vector<size_t> triIndices(1, sampleIDs[camParams.yRes / 2][camParams.xRes / 2] - 1);
				cout << "looking at tri index " << triIndices[0] << endl;
				const std::tuple<std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>, std::vector<uint8_t>> poseViewingTri = suggestPosesViewingMeshTriangles(curMapMesh, triIndices);
				if(std::get<1>(poseViewingTri)[0])
				{
					poseWrtCamToConsider = curCamPoseWrtMap.inverse() * std::get<0>(poseViewingTri)[0]; //looking-at-triangle pose wrt cur cam pose
					poseValid = true;
				}
			}

			if(!poseValid)
			{
				//use distance to camera as criterion instead
				poseWrtCamToConsider = rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, .4/* view dist; TODO ? */));

				const std::vector<float> poseDistancesFromCurPose = getPoseDistToCamPosesViewingMeshTriangles(poseWrtCamToConsider);

				/*
				 * set value of each triangle
				 */
				for(size_t i = 0; i < triangleIsSurface.size(); i++)
					if(!triangleIsSurface[i]) //if unseen
					{
						meshTriangleValues[i] = exp(-1/* TODO ? */ * poseDistancesFromCurPose[i]);
					}
			}
			else
			{
				/*
				 * set value of each triangle
				 */
				for(size_t i = 0; i < triangleIsSurface.size(); i++)
					if(!triangleIsSurface[i]) //if unseen
					{
						meshTriangleValues[i] = 1;
					}
			}

			break;
		}
		case triValueFuncType::DISTANCE_TO_MOVED_SURFACE:
		{
			const sumLogprobsCombined meshDiffingSums = getAggregatedDiffingResultsPerCurMeshTriangle(curMapMesh); //diffing info projected onto each cur-mesh triangles
			const std::vector<float> triSizes = getMeshTriangleSizes(*curMapMesh); //avg dist from centroid to a vertex, per triangle
			const std::vector<std::vector<size_t>> triNbrs = getTriangleNeighbors(*curMapMesh); //triangle index -> indices of nbring triangles, unsorted
			/*
			 * get distance from each tri to nearest high-p(m) tri using bfs on triangles (up to a max distance, for speed)
			 */
			const float maxDist = .03; //only find triangles up to this far (in m) from moved surfaces; TODO ?
			const std::vector<float> triGeodesicDistToMovedSurface = getUnseenTrianglesGeodesicDistToMovedSurface(curMapMesh, triSizes, triNbrs, meshDiffingSums, maxDist);

			/*
			 * set value of each triangle
			 */
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i]) //if unseen
				{
					if(triGeodesicDistToMovedSurface[i] == 0) meshTriangleValues[i] = 0; //don't value triangles we've already seen (these are the only ones that have distance 0 to surfaces known to be moved)
					else meshTriangleValues[i] = exp(-1/* TODO ? */ * std::min(triGeodesicDistToMovedSurface[i], 10.0f/* make this at least small enough to make sure we don't underflow */));
				}

			break;
		}
		case triValueFuncType::DISTANCE_HYBRID_TARGET_MOVED:
		{
			/*
			 * the location wrt the cur cam pose near which we want targets to be
			 *
			 * if we have a cur target defined, try to stay close to that; otherwise try to stay close to the cur cam pose
			 */
			const rgbd::eigen::Affine3f poseWrtCamToConsider = curTarget
																				? (framePosesWrtCurMap[frameIndex].inverse() * curTarget->camPoseWrtMap.get())
																				: rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, .4/* view dist; TODO ? */));
			const std::vector<float> poseDistancesFromTarget = getPoseDistToCamPosesViewingMeshTriangles(poseWrtCamToConsider);

			const sumLogprobsCombined meshDiffingSums = getAggregatedDiffingResultsPerCurMeshTriangle(curMapMesh); //diffing info projected onto each cur-mesh triangles
			const std::vector<float> triSizes = getMeshTriangleSizes(*curMapMesh); //avg dist from centroid to a vertex, per triangle
			const std::vector<std::vector<size_t>> triNbrs = getTriangleNeighbors(*curMapMesh); //triangle index -> indices of nbring triangles, unsorted
			/*
			 * get distance from each tri to nearest high-p(m) tri using bfs on triangles (up to a max distance, for speed)
			 */
			const float maxDist = .03; //only find triangles up to this far (in m) from moved surfaces; TODO ?
			const std::vector<float> triGeodesicDistToMovedSurface = getUnseenTrianglesGeodesicDistToMovedSurface(curMapMesh, triSizes, triNbrs, meshDiffingSums, maxDist);

			/*
			 * print stats about unseen triangles near high-p(m) triangles, for evaluating algorithms for completing objects segmented by change detection
			 */
			size_t numUnseenTrisNearMovedSurface = 0;
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i]) //if unseen
					if(triGeodesicDistToMovedSurface[i] < .04/* TODO ? */)
						numUnseenTrisNearMovedSurface++;
			cout << numUnseenTrisNearMovedSurface << " unseen tris near moved surfaces remain" << endl;

			/*
			 * set value of each triangle
			 */
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i]) //if unseen
				{
					meshTriangleValues[i] = exp(-1/* TODO ? */ * std::min(triGeodesicDistToMovedSurface[i], 10.0f/* make this at least small enough to make sure we don't underflow */) - .4/* TODO ? */ * poseDistancesFromTarget[i]);
				}

			break;
		}
		case triValueFuncType::DISTANCE_HYBRID_CAM_SURFACE:
		{
			const std::vector<size_t> surfaceTrianglesBorderingUnseen = findSurfaceTrianglesBorderingUnseen(curMapMesh, triangleIsSurface);
			const std::vector<float> triSizes = getMeshTriangleSizes(*curMapMesh); //avg dist from centroid to a vertex, per triangle
			const std::vector<std::vector<size_t>> triNbrs = getTriangleNeighbors(*curMapMesh); //triangle index -> indices of nbring triangles, unsorted

			/*
			 * get approx geodesic distance of each unseen tri to the surface
			 */
			rgbd::timer t;
			std::vector<float> triDistToSurface(curMapMesh->numTriangles(), -1); //approx dist in m along mesh surface; -1 = flag for unset
		{
			//bfs over triangles, adding approx distances; what we'll end up with is the sum, over a shortest path if you use length-1 edges, of approx nbr distances
			ASSERT_ALWAYS(onlineMapper->getCurSceneMapParams().volume_modeler.model_type == MODEL_SINGLE_VOLUME); //grids give you disconnected meshes, so geodesic distance will be screwed up
			std::unordered_set<size_t> finished;
			std::deque<size_t> toProcess; //tris have their distances set when they get added to this list
			for(size_t t : surfaceTrianglesBorderingUnseen) triDistToSurface[t] = 0;
			toProcess.insert(toProcess.end(), surfaceTrianglesBorderingUnseen.begin(), surfaceTrianglesBorderingUnseen.end());
			while(!toProcess.empty())
			{
				const size_t tri = toProcess.front();
				toProcess.pop_front();
				finished.insert(tri);
				for(size_t nbr : triNbrs[tri])
					if(triDistToSurface[nbr] < 0) //if it hasn't been added to toProcess yet
					{
						triDistToSurface[nbr] = triDistToSurface[tri] + triSizes[tri] + triSizes[nbr];
						toProcess.push_back(nbr);
					}
			}
		}
			t.stop("get geodesic mesh distances");

			/*
			 * get some proxy, in approximate meters, for the traversal cost from the current to each proposed cam pose
			 */
			t.restart();
			const rgbd::eigen::Affine3f poseWrtCamToConsider(rgbd::eigen::Translation3f(0, 0, .6/* view dist; TODO ? */)); //the location wrt the camera at which we want targets to be
			const std::vector<float> poseDistancesFromCurPose = getPoseDistToCamPosesViewingMeshTriangles(poseWrtCamToConsider, .1/* daWeight; TODO ? */);
			t.stop("get pose distances");

			/*
			 * set value of each triangle
			 */
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i]) //if unseen
				{
					meshTriangleValues[i] = exp(-.3/* TODO ? */ * triDistToSurface[i] - .3/* TODO ? */ * poseDistancesFromCurPose[i]);
				}

			break;
		}
		case triValueFuncType::DISTANCE_HYBRID_TARGET_SURFACE:
		{
			const std::vector<size_t> surfaceTrianglesBorderingUnseen = findSurfaceTrianglesBorderingUnseen(curMapMesh, triangleIsSurface);
			const std::vector<float> triSizes = getMeshTriangleSizes(*curMapMesh); //avg dist from centroid to a vertex, per triangle
			const std::vector<std::vector<size_t>> triNbrs = getTriangleNeighbors(*curMapMesh); //triangle index -> indices of nbring triangles, unsorted

			/*
			 * get approx geodesic distance of each unseen tri to the surface
			 */
			rgbd::timer t;
			std::vector<float> triDistToSurface(curMapMesh->numTriangles(), -1); //approx dist in m along mesh surface; -1 = flag for unset
		{
			//bfs over triangles, adding approx distances; what we'll end up with is the sum, over a shortest path if you use length-1 edges, of approx nbr distances
			ASSERT_ALWAYS(onlineMapper->getCurSceneMapParams().volume_modeler.model_type == MODEL_SINGLE_VOLUME); //grids give you disconnected meshes, so geodesic distance will be screwed up
			std::unordered_set<size_t> finished;
			std::deque<size_t> toProcess; //tris have their distances set when they get added to this list
			for(size_t t : surfaceTrianglesBorderingUnseen) triDistToSurface[t] = 0;
			toProcess.insert(toProcess.end(), surfaceTrianglesBorderingUnseen.begin(), surfaceTrianglesBorderingUnseen.end());
			while(!toProcess.empty())
			{
				const size_t tri = toProcess.front();
				toProcess.pop_front();
				finished.insert(tri);
				for(size_t nbr : triNbrs[tri])
					if(triDistToSurface[nbr] < 0) //if it hasn't been added to toProcess yet
					{
						triDistToSurface[nbr] = triDistToSurface[tri] + triSizes[tri] + triSizes[nbr];
						toProcess.push_back(nbr);
					}
			}
		}
			t.stop("get geodesic mesh distances");

			/*
			 * the location wrt the cur cam pose near which we want targets to be
			 *
			 * if we have a cur target defined, try to stay close to that; otherwise try to stay close to the cur cam pose
			 */
			const rgbd::eigen::Affine3f poseWrtCamToConsider = lastTarget
																				? (framePosesWrtCurMap[frameIndex].inverse() * lastTarget->camPoseWrtMap.get())
																				: rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, .5/* view dist; TODO ? */));
			const std::vector<float> poseDistancesFromTarget = getPoseDistToCamPosesViewingMeshTriangles(poseWrtCamToConsider, 0/*.1*//* daWeight; TODO ? */);

			/*
			 * set value of each triangle
			 */
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i]) //if unseen
				{
					meshTriangleValues[i] = exp(-.3/* TODO ? */ * triDistToSurface[i] - .3/* TODO ? */ * poseDistancesFromTarget[i]);
				}

			break;
		}
		case triValueFuncType::INFO_GAIN:
		{
			rgbd::timer t;
		//	const std::vector<size_t> surfaceTrianglesBorderingUnseen = findSurfaceTrianglesBorderingUnseen(curMapMesh, triangleIsSurface);

			/*
			 * get poses viewing all unseen mesh triangles
			 */
			t.restart();
			std::vector<size_t> unseenTriangleIndices;
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i])
					unseenTriangleIndices.push_back(i);
			std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> camPosesWrtMapViewingUnseenTris;
			std::vector<uint8_t> camPoseValidity;
			std::tie(camPosesWrtMapViewingUnseenTris, camPoseValidity) = suggestPosesViewingMeshTriangles(curMapMesh, unseenTriangleIndices);

			t.stop("suggest poses");

			t.restart();
			const sumLogprobsCombined meshDiffingSums; //doesn't need to be initialized if not used in rendering scoring

			/*
			 * approximate information gain by the number of unseen voxels visible in each view
			 */
			const std::vector<float> unseenVoxelCountScores = scorePosesByRendering(curMapMesh, triangleIsSurface, std::unordered_set<size_t>()/* unnec if not using to score *//*surfaceTrianglesBorderingUnseen*/, meshDiffingSums, 255, 0, 0, camPosesWrtMapViewingUnseenTris);
			const float maxScore = *std::max_element(unseenVoxelCountScores.begin(), unseenVoxelCountScores.end());
			t.stop("score poses by rendering");

			/*
			 * set value of each triangle
			 */
			for(size_t j = 0; j < unseenTriangleIndices.size(); j++)
				if(camPoseValidity[j]) //if we got a suggested pose viewing the tri
				{
					meshTriangleValues[unseenTriangleIndices[j]] = unseenVoxelCountScores[j] / maxScore;
				}

			break;
		}
		case triValueFuncType::TRAVERSAL_COST_FRONTIER:
		{
			/*
			 * plan to a pose viewing each frontier tri in turn, and compute path cost for each
			 */

			ASSERT_ALWAYS(robotInterface->robotIsFreeFlying()); //assert free-flying (this isn't designed for general-case robots)

			const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();

			//get poses viewing all mesh triangles
			std::vector<size_t> allTriangleIndices(curMapMesh->numTriangles());
			for(size_t i = 0; i < curMapMesh->numTriangles(); i++) allTriangleIndices[i] = i;
			std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> camPosesWrtMapViewingTris;
			std::vector<uint8_t> camPoseValidity;
			std::tie(camPosesWrtMapViewingTris, camPoseValidity) = suggestPosesViewingMeshTriangles(curMapMesh, allTriangleIndices);

			const std::vector<float> traversalCostByTriangle = getTraversalCostsToCamPoses(curMapMesh, curConfig, camPosesWrtMapViewingTris);

			/*
			 * set value of each triangle
			 */
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i]) //if unseen
				{
					meshTriangleValues[i] = exp(-1/* TODO ? */ * traversalCostByTriangle[i]);
				}

			break;
		}
		case triValueFuncType::DISTANCE_HYBRID_INFOGAIN_TRAVERSALCOST:
		{
			/*
			 * traversal cost
			 */

			ASSERT_ALWAYS(robotInterface->robotIsFreeFlying()); //assert free-flying (traversal cost isn't designed for general-case robots)

			const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();

			//get poses viewing all mesh triangles
			std::vector<size_t> allTriangleIndices(curMapMesh->numTriangles());
			for(size_t i = 0; i < curMapMesh->numTriangles(); i++) allTriangleIndices[i] = i;
			std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> camPosesWrtMapViewingTris;
		{
			std::vector<uint8_t> camPoseValidity;
			std::tie(camPosesWrtMapViewingTris, camPoseValidity) = suggestPosesViewingMeshTriangles(curMapMesh, allTriangleIndices);
		}

			const std::vector<float> traversalCostByTriangle = getTraversalCostsToCamPoses(curMapMesh, curConfig, camPosesWrtMapViewingTris);

			/*
			 * info gain
			 */

			/*
			 * get poses viewing all unseen mesh triangles
			 */
			t.restart();
			std::vector<size_t> unseenTriangleIndices;
			for(size_t i = 0; i < triangleIsSurface.size(); i++)
				if(!triangleIsSurface[i])
					unseenTriangleIndices.push_back(i);
			std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> camPosesWrtMapViewingUnseenTris;
			std::vector<uint8_t> camPoseValidityUnseen;
			std::tie(camPosesWrtMapViewingUnseenTris, camPoseValidityUnseen) = suggestPosesViewingMeshTriangles(curMapMesh, unseenTriangleIndices);

			t.stop("suggest poses");

			t.restart();
			const sumLogprobsCombined meshDiffingSums; //doesn't need to be initialized if not used in rendering scoring

			/*
			 * approximate information gain by the number of unseen voxels visible in each view
			 */
			const std::vector<float> unseenVoxelCountScores = scorePosesByRendering(curMapMesh, triangleIsSurface, std::unordered_set<size_t>()/* unnec if not using to score *//*surfaceTrianglesBorderingUnseen*/, meshDiffingSums, 255, 0, 0, camPosesWrtMapViewingUnseenTris);
			const float maxScore = *std::max_element(unseenVoxelCountScores.begin(), unseenVoxelCountScores.end());
			t.stop("score poses by rendering");

			/*
			 * set value of each triangle
			 */
			const float alpha = .6; //tradeoff parameter btwn the two distances; TODO ?
			for(size_t j = 0; j < unseenTriangleIndices.size(); j++)
				if(camPoseValidityUnseen[j]) //if we got a suggested pose viewing the tri
				{
					meshTriangleValues[unseenTriangleIndices[j]] = alpha * unseenVoxelCountScores[j] / maxScore + (1 - alpha) * exp(-1/* TODO ? */ * traversalCostByTriangle[unseenTriangleIndices[j]]);
				}

			break;
		}
	}
	t.stop("set triangle values");

#if 0 //visualize tri value
	static int c = 0;
	if(c % 2 == 0) //keep disk space requirements down
{
	t.restart();
	const float maxTriValue = *std::max_element(meshTriangleValues.begin(), meshTriangleValues.end());
	std::shared_ptr<triangulatedMesh> vismesh(new triangulatedMesh);
	vismesh->allocateVertices(3 * curMapMesh->numTriangles());
	vismesh->allocateTriangles(curMapMesh->numTriangles());
	const auto& tris = curMapMesh->getTriangles();
	for(size_t i = 0; i < curMapMesh->numTriangles(); i++)
	{
		for(size_t k = 0; k < 3; k++)
		{
			rgbd::pt v = curMapMesh->v(tris[i].v[k]);
			v.rgb = rgbd::packRGB(255, 255 * std::max(0.0f, meshTriangleValues[i]) / maxTriValue, 0);
			vismesh->setVertex(i * 3 + k, v);
		}
		vismesh->setTriangle(i, triangulatedMesh::triangle{{{(uint32_t)i * 3 + 0, (uint32_t)i * 3 + 1, (uint32_t)i * 3 + 2}}});
	}

	std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWrtMapWithScores;
	visualizeMeshWithCameraPoses(*vismesh, camPosesWrtMapWithScores, rgbd::eigen::Affine3f::Identity()/* viewingPoseWrtMap */, true/* show cur pose */,
		"", outdir / (boost::format("trivalues%1%.ply") % frameIndex).str());
	t.stop("visualize tri values");
}
	c++;
#endif

	/********************************************************************************************************************************************************
	 * precompute stuff to be used in suggestMoreCamPoses()
	 */

	if(geometricEnvData) data->collisionCheckingEnvData = geometricEnvData;
	else
	{
		t.restart();
		data->collisionCheckingEnvData = precomputeEnvironmentGeometryForCollisionChecking();
		t.stop("precompute geometry for coll check");
	}

	/********************************************************************************************************************************************************
	 * suggest poses to score
	 */
{
	rgbd::timer t;
#if 1
	data->suggestedTargets = std::move(suggestOrderedViewTargets(curMapPoseWrtRaveWorld, curMapMesh, meshTriangleValues));
#define SUGGESTED_POSES_ARE_ALREADY_ORDERED
#elif 1
	data->suggestedPoses = std::move(suggestPosesNearCurrent(curMapPoseWrtRaveWorld));
#elif 1
	data->suggestedPoses = std::move(suggestPosesViewingNearestFrontierTriangle(curMapPoseWrtRaveWorld, curMapMesh, frontierTrianglesVec));
#define SUGGESTED_POSES_ARE_ALREADY_ORDERED
#endif
	data->numTargetsProcessedSoFar = 0;
	data->numPosesProcessedSoFar = 0;
	data->numConfigsProcessedSoFar = 0;
	data->numReachablePosesSoFar = 0;
	data->numNoncollidingConfigsSoFar = 0;
	t.stop("call suggestPosesViewingValuableTriangles");
}
	cout << "listed " << data->suggestedTargets.size() << " possible targets" << endl;
	t2.stop("run initCamPoseSuggestion");
	return data;
}

/*
 * return more camera poses/robot configurations to try planning to, or empty if we're out of ideas
 *
 * return no more than maxSuggestionsAtOnce suggestions at a time (this function is meant to be called until an acceptable suggestion is found)
 */
std::vector<onlineActiveModeler::suggestedCamPoseInfo> onlineActiveModeler::suggestMoreCamPoses(const std::shared_ptr<camPoseSuggestionData>& data, const size_t maxSuggestionsAtOnce)
{
	rgbd::timer t;

	static size_t callNum = 0; //gets reset at the beginning of processing a new frame (when frameIndex increases)
	static int64_t lastFrameIndex = -3; //the last frameIndex with which we were called
	if(frameIndex != lastFrameIndex)
	{
		callNum = 0;
		lastFrameIndex = frameIndex;
	}
	else
	{
		callNum++;
	}

	std::vector<onlineActiveModeler::suggestedCamPoseInfo> suggestedPoses;
	const OpenRAVE::RobotBasePtr planningRobot = planningEnv->GetRobot(robotCamHandler->getRobotName());

	while(suggestedPoses.empty() && data->numTargetsProcessedSoFar < data->suggestedTargets.size())
	{
		//copy out the few cam poses we'll process this time
		//TODO right now it doesn't necessarily limit itself to maxSuggestions; replace the max parameter with a hint, or actually use it like a max, or something
		while(suggestedPoses.size() < maxSuggestionsAtOnce && data->numTargetsProcessedSoFar < data->suggestedTargets.size())
		{
			const std::vector<std::vector<OpenRAVE::dReal>> viewingConfigurations = std::move(robotInterface->getViewingConfigurationsForTarget(planningRobot/* TODO does it matter which robot I use? */, robotCamHandler, curMapPoseWrtRaveWorld * data->suggestedTargets[data->numTargetsProcessedSoFar].tgtPoseWrtMap));
			const size_t prevPosesSize = suggestedPoses.size();
			suggestedPoses.resize(suggestedPoses.size() + viewingConfigurations.size());
			for(size_t i = 0; i < viewingConfigurations.size(); i++)
			{
				suggestedPoses[prevPosesSize + i].targetInfo = data->suggestedTargets[data->numTargetsProcessedSoFar];
				suggestedPoses[prevPosesSize + i].camPoseWrtMap = curMapPoseWrtRaveWorld.inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(viewingConfigurations[i]) * robotCamHandler->getCamPoseWrtRobotBase(viewingConfigurations[i]);
				suggestedPoses[prevPosesSize + i].viewingConfiguration = viewingConfigurations[i];
			}
			data->numTargetsProcessedSoFar++;
		}
		data->numPosesProcessedSoFar += suggestedPoses.size(); //TODO is this right?
		data->numConfigsProcessedSoFar += suggestedPoses.size();

#ifdef DEBUG_POSE_SUGGESTION //visualize all suggested poses
	t.restart();
{
	std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWrtMapWithScores(suggestedPoses.size());
	for(size_t i = 0; i < suggestedPoses.size(); i++)
	{
		camPosesWrtMapWithScores[i].first = suggestedPoses[i].camPoseWrtMap.get();

		float dx, da;
		xf::transform_difference(rgbd::eigen::Affine3f::Identity(), camPosesWrtMapWithScores[i].first, dx, da);
		camPosesWrtMapWithScores[i].second = std::min(1.0f, dx / 4);
	}
	visualizeMeshWithCameraPoses(*curMapMeshSeenOnly, camPosesWrtMapWithScores, framePosesWrtCurMap[0] * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -1))/* viewingPoseWrtMap */, true/* show cur pose */,
		fs::path(), outdir / (boost::format("shortlistPoses%1%-%2%.ply") % frameIndex % callNum).str());
}
	t.stop("visualize suggested poses");
#endif

	/********************************************************************************************************************************************************
	 * filter poses to score by reachability
	 */
	if(robotCamHandler)
	{
		std::vector<uint8_t> reachabilityFlags(suggestedPoses.size(), true); //whether each pose is reachable
#if 0 //20140506 now calling getViewingConfigurationsForTarget() above
		/********************************************************************************************************************************************************
		 * reachability possibly without collision checking
		 *
		 * if doing coll checking here: with the bullet collision checker, even with 14 threads this takes .95s; ode even w/ one thread is .65s for the same # of poses
		 */
		t.restart();
		std::vector<std::vector<std::vector<OpenRAVE::dReal>>> viewingConfigurations; //target -> zero or more configs viewing target
	{
		std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> camPosesWrtMap(suggestedPoses.size());
		for(size_t i = 0; i < suggestedPoses.size(); i++) camPosesWrtMap[i] = suggestedPoses[i].camPoseWrtMap;
#ifndef USE_CGAL_COLLISION_CHECKING
		viewingConfigurations = std::move(robotInterface->getViewingConfigurationsForTargets(?curMapPoseWrtRaveWorld, camPosesWrtMap, true/* collision checking */));
		for(size_t i = 0; i < camPosesWrtMap.size(); i++)
			if(!viewingConfigurations[i].empty())
				reachabilityFlags[i] = true;
#else
		viewingConfigurations = std::move(robotInterface->getViewingConfigurationsForTargets(?curMapPoseWrtRaveWorld, camPosesWrtMap, false/* collision checking */));
#endif
	}

	size_t numReachable = 0;
	for(size_t i = 0; i < suggestedPoses.size(); i++)
		if(!viewingConfigurations[i].empty())
			numReachable++;
	data->numReachableSoFar += numReachable;
	cout << numReachable << " poses are reachable (total: " << data->numReachableSoFar << " of " << data->numPosesProcessedSoFar << ")" << endl;

#ifdef DEBUG_POSE_SUGGESTION //debugging
{
	std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWrtMapWithScores(numReachable);
	for(size_t i = 0, j = 0; i < suggestedPoses.size(); i++)
		if(!viewingConfigurations[i].empty())
		{
			camPosesWrtMapWithScores[j].first = suggestedPoses[i].camPoseWrtMap;
			camPosesWrtMapWithScores[j].second = 1;
			j++;
		}
	visualizeMeshWithCameraPoses(*curMapMeshSeenOnly, camPosesWrtMapWithScores, framePosesWrtCurMap[0] * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -1))/* viewingPoseWrtMap */, true/* show cur pose */,
		outdir / (boost::format("reachablePoses%1%.png") % frameIndex).str());
}
#endif
		t.stop("check reachability w/o collision");
#endif
		/************************************************************************************************************************************************
		 * reachability with collision checking
		 */
		t.restart();
		std::vector<std::vector<OpenRAVE::dReal>> viewingConfigurations(suggestedPoses.size());
		for(size_t i = 0; i < suggestedPoses.size(); i++) viewingConfigurations[i] = suggestedPoses[i].viewingConfiguration;
		const std::vector<uint8_t> collisionFlags = checkRobotCollisions(data->collisionCheckingEnvData, viewingConfigurations);
		for(size_t i = 0; i < viewingConfigurations.size(); i++)
			if(!viewingConfigurations[i].empty()) //if collision-check result is meaningful
				reachabilityFlags[i] = !collisionFlags[i];
		t.stop("check next pose collision");

		t.restart();
		std::vector<onlineActiveModeler::suggestedCamPoseInfo> reachableSuggestedPoses;
		copySelected(suggestedPoses, reachabilityFlags, std::back_inserter(reachableSuggestedPoses));
		data->numNoncollidingConfigsSoFar += reachableSuggestedPoses.size();
		cout << reachableSuggestedPoses.size() << " of " << suggestedPoses.size() << " configs are noncolliding (total: " << data->numNoncollidingConfigsSoFar << " of " << data->numConfigsProcessedSoFar << ")" << endl;
		suggestedPoses = std::move(reachableSuggestedPoses);
		t.stop("copy filtered poses");
	}

#ifndef SUGGESTED_POSES_ARE_ALREADY_ORDERED
	/*
	 * score poses by rendering
	 */
	t.restart();
	camPosesWrtMapWithScores = std::move(scorePosesByRendering(curMapMesh, triangleIsSurface, surfaceTrianglesBorderingUnseen, , , , camPosesWrtMap));
	t.stop("score all views");
	std::stable_sort(camPosesWrtMapWithScores.begin(), camPosesWrtMapWithScores.end(), [](const std::pair<rgbd::eigen::Affine3f, float>& s1, const std::pair<rgbd::eigen::Affine3f, float>& s2){return s1.second > s2.second;});
#else
#if 0 //20140506 score is now not part of the pose suggestion struct
	/*
	 * give all poses the same arbitrary score, but preserve their order
	 */
	for(auto& s : suggestedPoses) s.score = 0;
#endif
#endif

#ifdef DEBUG_POSE_SUGGESTION //visualize scored poses
	t.restart();
{
	const size_t numPosesToShow = std::min(suggestedPoses.size(), (size_t)20);
	std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWithScoresForVis(numPosesToShow);
	for(size_t i = 0; i < numPosesToShow; i++)
	{
		camPosesWithScoresForVis[i].first = suggestedPoses[i].camPoseWrtMap;

		camPosesWithScoresForVis[i].second = (i == 0) ? 1 : .5 * (suggestedPoses[i].score - suggestedPoses[numPosesToShow - 1].score) / (suggestedPoses[0].score - suggestedPoses[numPosesToShow - 1].score);
	}
	visualizeMeshWithCameraPoses(*curMapMeshSeenOnly, camPosesWithScoresForVis, framePosesWrtCurMap[0] * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -1))/* viewingPoseWrtMap */, true/* show cur pose */,
		outdir / (boost::format("scoredReachablePoses%1%.png") % frameIndex).str());
	//visualizeRobotPoses(curMapMesh, viewingConfigurations, outdir / (boost::format("scoredReachablePose%1%_") % frameIndex).str());
}
	t.stop("visualize scored poses");
#endif
	}

	return suggestedPoses;
}

onlineActiveModeler::poseQuantizationIndex onlineActiveModeler::getPoseQuantizationIndex(const std::shared_ptr<triangulatedMesh>& curMapMesh, const suggestedCamPoseInfo& suggestedPose, const VolumeModelerAllParams& mapperParams) const
{
	poseQuantizationIndex index;
	/*
	 * decide which pose quantization bin this pose is in
	 */
	const rgbd::eigen::Vector3f tgtTriCentroid = (rgbd::ptX2eigen<rgbd::eigen::Vector3f>(suggestedPose.targetInfo.triVertices[0]) + rgbd::ptX2eigen<rgbd::eigen::Vector3f>(suggestedPose.targetInfo.triVertices[1]) + rgbd::ptX2eigen<rgbd::eigen::Vector3f>(suggestedPose.targetInfo.triVertices[2])) / 3;
	for(size_t k = 0; k < 3; k++) index.voxel[k] = rint((tgtTriCentroid[k] - .5 * mapperParams.volume.cell_size) / mapperParams.volume.cell_size);
	const rgbd::eigen::Vector3f viewVec = suggestedPose.camPoseWrtMap.get().linear() * rgbd::eigen::Vector3f(0, 0, 0) - tgtTriCentroid;
	if(fabs(viewVec[0]) > std::max(fabs(viewVec[1]), fabs(viewVec[2])))
	{
		if(viewVec[0] > 0) index.ori = viewsByPoseOri::POSX;
		else index.ori = viewsByPoseOri::NEGX;
	}
	else if(fabs(viewVec[1]) > fabs(viewVec[2]))
	{
		if(viewVec[1] > 0) index.ori = viewsByPoseOri::POSY;
		else index.ori = viewsByPoseOri::NEGY;
	}
	else
	{
		if(viewVec[2] > 0) index.ori = viewsByPoseOri::POSZ;
		else index.ori = viewsByPoseOri::NEGZ;
	}
	return index;
}

/****************************************************************************************************************************************************************************************************************************************/

/*
 * compute signed distance fields for all objects for orcdchomp
 */
void onlineActiveModeler::computeSDFsForCHOMP()
{
	std::ostringstream outstr;
	std::stringstream args;

	std::vector<OpenRAVE::KinBodyPtr> kinbodies;
	planningEnv->GetBodies(kinbodies);
	for(const auto& b : kinbodies)
		if(b->GetName() != robotCamHandler->getRobotName()
#ifdef MAKE_MAP_MESH_SDF_OURSELVES
			&& b->GetName() != "mapMesh"
#endif
			&& b->GetName().find("nochomp") == std::string::npos //in-name flag to have kinbodies ignored by chomp
			&& b->IsEnabled())
		{
			//when using computedistancefield, all bodies other than the one being computed for must be disabled
			for(const auto& b2 : kinbodies)
				if(b2 != b)
					b2->Enable(false);

			cout << "computingdistancefield " << b->GetName() << endl;
			args << "computedistancefield kinbody " << b->GetName();
			if(!chompModule->SendCommand(outstr, args))
			{
				cout << "computedistancefield '" << b->GetName() << "' failed" << endl;
				ASSERT_ALWAYS(false);
			}

			for(const auto& b2 : kinbodies)
				if(b2 != b)
					b2->Enable(true);
		}

#ifdef MAKE_MAP_MESH_SDF_OURSELVES
	/*
	 * create an sdf for the map mesh ourselves (much faster than giving chomp the openrave kinbody, but requires special handling with grid maps, which is why to maybe not use it)
	 */
	const VolumeModelerAllParams& curSceneMapParams = onlineMapper->getCurSceneMapParams();
	rgbd::timer t2, t;

	orcdchomp::sdf chompSDF;
	strcpy(chompSDF.kinbody_name, "mapMesh");
	/* set pose of grid w.r.t. kinbody (here a mesh) frame */
	const auto& M = rgbd::eigen::Affine3f::Identity();//curMapPoseWrtRobotBase; //the pose we want
	double xlation[3] = {M(0, 3), M(1, 3), M(2, 3)};
	double rotMtx[3][3] =
	{
		{M(0, 0), M(0, 1), M(0, 2)},
		{M(1, 0), M(1, 1), M(1, 2)},
		{M(2, 0), M(2, 1), M(2, 2)}
	};
	cd_kin_pose_from_dR(chompSDF.pose, xlation, rotMtx);
//	cd_mat_vec_print("kin pose: ", chompSDF.pose, 7); //20131125 this looks right (doesn't match chomp's exactly because of their padding and such)

	std::array<int, 3> gsdf_sizearray;
	for(size_t i = 0; i < 3; i++) gsdf_sizearray[i] = curSceneMapParams.volume.cell_count[i];
	double defaultValue = 1; //1 = free space -- see orcdchomp

//#define GET_SDF_FROM_CHOMP //chomp is much slower than Evan's gpu code here
#ifdef GET_SDF_FROM_CHOMP
	t.restart();
	cd_grid* g_obs = NULL;
	const int err = cd_grid_create_sizearray(&g_obs, &defaultValue, sizeof(double), 3, gsdf_sizearray.data());
	if(err) throw std::runtime_error("Not enough memory for distance field!");

	/*
	 * copy tsdf to occupancy grid
	 */
	t.restart();
{
	for(size_t i = 0; i < 3; i++) g_obs->lengths[i] = curSceneMapParams.volume.cell_count[i] * curSceneMapParams.volume.cell_size;
	//fill occupancy grid: HUGE_VAL for occupied, 0 for free
	double* g = reinterpret_cast<double*>(g_obs->data);
	//cd_grid is indexed by (x, y, z)
	for(size_t ix = 0; ix < (size_t)gsdf_sizearray[0]; ix++)
		for(size_t iy = 0; iy < (size_t)gsdf_sizearray[1]; iy++)
			for(size_t iz = 0; iz < (size_t)gsdf_sizearray[2]; iz++, g++)
			{
				//the tsdf value is a metric distance, and the truncation cutoff is >= 3 * voxelSize, so can use any threshold < 3 * voxelSize here
				if(bufferDVector[(iz * curSceneMapParams.volume.cell_count[1] + iy) * curSceneMapParams.volume.cell_count[0] + ix] < 1.5 * curSceneMapParams.volume.cell_size)
					*g = HUGE_VAL;
				else
					*g = 0;
			}
}
	t.stop("fill chomp grid");
#if 0 //visualize occupancy
{
	voxelGrid<> grid;
	grid.resolution = .01;
	grid.voxels.resize(boost::extents[gsdf_sizearray[2]][gsdf_sizearray[1]][gsdf_sizearray[0]]);
	for(size_t i = 0; i < g_obs->ncells; i++)
	{
		std::array<int, 3> voxelIndices;
		cd_grid_index_to_subs(g_obs, i, voxelIndices.data());
		const double v = *(double*)cd_grid_get_index(g_obs, i);
		grid(voxelIndex{{{(uint16_t)voxelIndices[0], (uint16_t)voxelIndices[1], (uint16_t)voxelIndices[2]}}}).value = (v == HUGE_VAL) ? voxels::FREE_VALUE : 32768;
	}
	vrip::writeRLEVRI(grid, (boost::format("sdf%1%.vri") % frameIndex).str());
	cout << "wrote grid" << endl;
}
#endif

	t.restart();
	/*
	 * have chomp create an sdf voxel grid from the binary occupancy grid
	 */
	cd_grid_double_bin_sdf(&chompSDF.grid, g_obs);
	cd_grid_destroy(g_obs);
	t.stop("compute sdf");
#else //cuda sdf computation
{
	rgbd::timer t;
	voxelGrid<uint8_t> occGrid;
	occGrid.resolution = curSceneMapParams.volume.cell_size;
	occGrid.voxels.resize(boost::extents[curSceneMapParams.volume.cell_count[2]][curSceneMapParams.volume.cell_count[1]][curSceneMapParams.volume.cell_count[0]]);
	uint8_t* o = occGrid.voxels.data();
	const std::vector<float>& bufferDVector = *bufferDVectors.at(0);
	for(size_t l = 0; l < occGrid.voxels.num_elements(); l++, o++)
		*o = (bufferDVector[l] < 1.5 * curSceneMapParams.volume.cell_size) ? 0 : 1;
	t.stop("fill occupancy grid for cuda");
	t.restart();
	voxelGrid<float> distXformGrid;
	computeSignedDistanceFieldFromOccupancy(occGrid, 0, distXformGrid, true/* reverse memory ordering of result */); //input is in [z][y][x] memory order; output is in [x][y][z]
	t.stop("compute sdf w/ cuda");
#if 0 //debug: compare to chomp output (20131203 it matches)
	for(size_t iz = 0; iz < (size_t)curSceneMapParams.volume.cell_count[2]; iz++)
		for(size_t iy = 0; iy < (size_t)curSceneMapParams.volume.cell_count[1]; iy++)
			for(size_t ix = 0; ix < (size_t)curSceneMapParams.volume.cell_count[0]; ix++)
				if(fabs(distXformGrid.voxels[ix][iy][iz] - *((double*)chompSDF.grid->data + (ix * gsdf_sizearray[1] + iy) * gsdf_sizearray[2] + iz)) > 1e-4)
				{
					cout << "mismatch " << iz << ' ' << iy << ' ' << ix << ": " << distXformGrid.voxels[ix][iy][iz] << ", " << *((double*)chompSDF.grid->data + (ix * gsdf_sizearray[1] + iy) * gsdf_sizearray[2] + iz) << endl;
				}
	cout << "waiting" << endl;
	int q; std::cin >> q;
#endif

	/*
	 * copy to chomp structure
	 */
	t.restart();
	const int err = cd_grid_create_sizearray(&chompSDF.grid, &defaultValue, sizeof(double), 3, gsdf_sizearray.data());
	for(size_t k = 0; k < 3; k++) chompSDF.grid->lengths[k] = curSceneMapParams.volume.cell_count[k] * curSceneMapParams.volume.cell_size;
	ASSERT_ALWAYS(!err);
	//TODO multithread the copy? takes .16s
	std::copy(distXformGrid.voxels.data(), distXformGrid.voxels.data() + distXformGrid.voxels.num_elements(), (double*)chompSDF.grid->data); //distXformGrid is now in chomp ([x][y][z]) memory order
	t.stop("copy sdf to chomp");
}
#endif
	/*
	 * give chomp responsibility for the sdf structure's memory
	 */
	args << "computedistancefield binary_occ_grid_addr " << &chompSDF;
	if(!chompModule->SendCommand(outstr, args))
	{
		cout << "computedistancefield w/ sdf passed failed" << endl;
		ASSERT_ALWAYS(false);
	}
	t2.stop("manually create sdf");
#endif
}

/*
 * remove stored signed distance fields from orcdchomp
 */
void onlineActiveModeler::clearSDFsFromCHOMP()
{
	std::ostringstream outstr;
	cout << "calling clear_sdfs" << endl;
{
	std::stringstream args;
	args << "clear_sdfs";
	if(!chompModule->SendCommand(outstr, args))
	{
		ASSERT_ALWAYS(false && "clear_sdfs failed");
	}
}

}

/*
 * resample the traj so the temporal and spatial distance between each pair of waypts is what we think mapping can handle in one iteration (in order to not break the small-movement assumption)
 *
 * deltaT: time taken between each consecutive pair of samples
 */
void onlineActiveModeler::retimeTrajectoryForMapping(std::vector<std::vector<OpenRAVE::dReal>>& trajSamples, OpenRAVE::dReal& deltaT)
{
	if(inSimulation)
	{
		/*
		 * resample the interval between each pair of waypts separately
		 */
		std::vector<size_t> numResampledWayptsPerPair;
		size_t numResampledWaypts = 0;
		for(size_t i = 0; i < trajSamples.size() - 1; i++) //for each waypt pair
		{
			const rgbd::eigen::Affine3f camLinkPoseWrtRaveWorldBefore = robotCamHandler->getRobotBasePoseWrtRaveWorld(trajSamples[i]) * robotCamHandler->getCamAttachmentLinkPoseWrtRobotBase(trajSamples[i]),
				camLinkPoseWrtRaveWorldAfter = robotCamHandler->getRobotBasePoseWrtRaveWorld(trajSamples[i + 1]) * robotCamHandler->getCamAttachmentLinkPoseWrtRobotBase(trajSamples[i + 1]);
			float dx, da;
			xf::transform_difference(camLinkPoseWrtRaveWorldBefore, camLinkPoseWrtRaveWorldAfter, dx, da);
			const float poseDist = dx + mappingMaxIterationDX / mappingMaxIterationDA * da;
			numResampledWayptsPerPair.push_back(std::max((size_t)1, (size_t)ceil(poseDist / mappingMaxIterationDX)) + 1); //these won't sum to numResampledWaypts since we include duplicates at consecutive interval end/begin in this count
			numResampledWaypts += numResampledWayptsPerPair.back() - ((i == trajSamples.size() - 2) ? 0 : 1); //don't duplicate waypoints at end/begin of consecutive subintervals
		}

		std::vector<std::vector<OpenRAVE::dReal>> newTrajSamples(numResampledWaypts);
		for(size_t i = 0, l = 0; i < trajSamples.size() - 1; i++) //for each waypt pair
		{
			for(size_t j = 0; j < numResampledWayptsPerPair[i] - ((i == trajSamples.size() - 2) ? 0 : 1)/* don't duplicate waypoints at end/begin of consecutive subintervals */; j++, l++)
			{
				const float alpha = (float)j / (numResampledWayptsPerPair[i] - 1); //linterp parameter
				newTrajSamples[l] = robotInterface->linterpConfigurations(trajSamples[i], trajSamples[i + 1], alpha);
			}
		}

		trajSamples = std::move(newTrajSamples);
		deltaT = estimatedMappingIterationRuntime;
	}
	else
	{
		float largestWayptPoseDist = 0;
		for(size_t i = 0; i < trajSamples.size() - 1; i++)
		{
			const rgbd::eigen::Affine3f camLinkPoseWrtRaveWorldBefore = robotCamHandler->getRobotBasePoseWrtRaveWorld(trajSamples[i]) * robotCamHandler->getCamAttachmentLinkPoseWrtRobotBase(trajSamples[i]),
				camLinkPoseWrtRaveWorldAfter = robotCamHandler->getRobotBasePoseWrtRaveWorld(trajSamples[i + 1]) * robotCamHandler->getCamAttachmentLinkPoseWrtRobotBase(trajSamples[i + 1]);
			float dx, da;
			xf::transform_difference(camLinkPoseWrtRaveWorldBefore, camLinkPoseWrtRaveWorldAfter, dx, da);
			const float poseDist = dx + mappingMaxIterationDX / mappingMaxIterationDA * da;
			if(poseDist > largestWayptPoseDist) largestWayptPoseDist = poseDist;
		}
		deltaT = largestWayptPoseDist / mappingMaxIterationDX * estimatedMappingIterationRuntime; //scale so the fastest waypt-pair traversal is still slow enough for mapping to not break
	}
}

/*
 * read tsdf values
 *
 * return the value for unseen if pos is outside map bounds
 */
float onlineActiveModeler::tsdfValueAtPos(const rgbd::eigen::Vector3f& posWrtRaveWorld) const
{
	const VolumeModelerAllParams& curSceneMapParams = onlineMapper->getCurSceneMapParams();
	const rgbd::eigen::Vector3f posWrtMap = curMapPoseWrtRaveWorld.inverse() * posWrtRaveWorld;
	const float voxelSize = curSceneMapParams.volume.cell_size;
	if(curSceneMapParams.volume_modeler.model_type == MODEL_SINGLE_VOLUME)
	{
		for(size_t k = 0; k < 3; k++)
			if(posWrtMap[k] < 0 || posWrtMap[k] > voxelSize * curSceneMapParams.volume.cell_count[k]) //if out of map bounds
				return -1; //unseen

		//return state at nearest voxel
		rgbd::eigen::Vector3i nearestVoxel;
		for(size_t k = 0; k < 3; k++) nearestVoxel[k] = std::max((int)0, std::min((int)curSceneMapParams.volume.cell_count[k] - 1, (int)floor(posWrtMap[k] / voxelSize)));
		const size_t voxelIndex = (nearestVoxel[2] * curSceneMapParams.volume.cell_count[1] + nearestVoxel[1]) * curSceneMapParams.volume.cell_count[0] + nearestVoxel[0];
		return (*bufferDVectors[0])[voxelIndex];
	}
	else
	{
		const size_t gridVolumeWidth = curSceneMapParams.grid.grid_size + 2 * curSceneMapParams.grid.border_size + 1; //the +1 is something peter did to make meshes look nicer (better overlap of grid volumes)
		const int32_t gridVolumeDimMin = -curSceneMapParams.grid.border_size, gridVolumeDimMax = gridVolumeDimMin + gridVolumeWidth; //edges of each volume in each dimension, as a multiple of voxel size
		for(size_t i = 0; i < bufferDVectors.size(); i++) //for each grid volume
		{
			const rgbd::eigen::Vector3f posWrtVolume = tsdfBufferPosesWrtMap[i]->inverse() * posWrtMap;
			if(posWrtVolume[0] >= voxelSize * gridVolumeDimMin && posWrtVolume[0] < voxelSize * gridVolumeDimMax //if the point's within this grid cell
				&& posWrtVolume[1] >= voxelSize * gridVolumeDimMin && posWrtVolume[1] < voxelSize * gridVolumeDimMax
				&& posWrtVolume[2] >= voxelSize * gridVolumeDimMin && posWrtVolume[2] < voxelSize * gridVolumeDimMax)
			{
				//return state at nearest voxel
				rgbd::eigen::Vector3i nearestVoxel;
				for(size_t k = 0; k < 3; k++) nearestVoxel[k] = std::max((int)0, std::min((int)curSceneMapParams.grid.grid_size - 1, (int)floor(posWrtVolume[k] / voxelSize))) + curSceneMapParams.grid.border_size;
				const size_t voxelIndex = (nearestVoxel[2] * gridVolumeWidth + nearestVoxel[1]) * gridVolumeWidth + nearestVoxel[0];
				return (*bufferDVectors[i])[voxelIndex];
			}
		}
		return -1; //unseen
	}
}
/*
 * read tsdf values
 *
 * return whether the map says the given position is known free (this includes a check for it being outside map bounds)
 */
bool onlineActiveModeler::tsdfFreeAtPos(const rgbd::eigen::Vector3f& posWrtRaveWorld) const
{
	const float voxelSize = onlineMapper->getCurSceneMapParams().volume.cell_size;
	return tsdfValueAtPos(posWrtRaveWorld) > 2 * voxelSize; //TODO a better way? in particular this doesn't ensure we're at least 2*voxelsize distance from surfaces in unseen regions--just from seen ones
}

/*
 * return whether we're currently (i.e. on the current frame) using the CHOMP motion planner
 */
bool onlineActiveModeler::usingChomp() const
{
#ifdef PLAN_WITH_CHOMP
	return robotInterface->robotIsKinematic();
#else
	return false;
#endif
}

/*
 * plan a motion using simple configuration interpolation
 *
 * ideally usable with any type of robot
 *
 * intended to be used only for small motions
 */
onlineActiveModeler::motionPlanInfo onlineActiveModeler::planPathByConfigInterpolation(const std::vector<OpenRAVE::dReal>& planStartConfig, const std::vector<OpenRAVE::dReal>& targetConfig)
{
	motionPlanInfo plan;

	const rgbd::eigen::Affine3f startCamPoseWrtRaveWorld = robotCamHandler->getRobotBasePoseWrtRaveWorld(planStartConfig) * robotCamHandler->getCamPoseWrtRobotBase(planStartConfig);
	const rgbd::eigen::Affine3f targetCamPoseWrtRaveWorld = robotCamHandler->getRobotBasePoseWrtRaveWorld(targetConfig) * robotCamHandler->getCamPoseWrtRobotBase(targetConfig);
	const float daFactor = mappingMaxIterationDX / mappingMaxIterationDA;
	float dx, da;
	xf::transform_difference(startCamPoseWrtRaveWorld, targetCamPoseWrtRaveWorld, dx, da);
	const float targetDistFromStart = dx + daFactor * da;
	const float pathTraversalDist = estimatedTrajPlanTime / estimatedMappingIterationRuntime * mappingMaxIterationDX; //how far we think we can get in the time it takes to plan
	const float alpha = (pathTraversalDist >= targetDistFromStart) ? 1 : pathTraversalDist / targetDistFromStart; //percentage of distance to target that we'll move in this plan
	cout << "[plan] distToMove " << pathTraversalDist << ", targetDist " << targetDistFromStart << endl;
	cout << "[plan] alpha " << alpha << endl;
	plan.reachesTarget = (alpha >= 1);
	const std::vector<OpenRAVE::dReal> planEndConfig = robotInterface->linterpConfigurations(planStartConfig, targetConfig, alpha);
	plan.waypoints.resize(2);
	plan.waypoints[0] = planStartConfig;
	plan.waypoints[1] = planEndConfig;
	plan.deltaT = alpha * estimatedTrajPlanTime;
	cout << "[plan] start "; std::copy(planStartConfig.begin(), planStartConfig.end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
	cout << "[plan] end "; std::copy(planEndConfig.begin(), planEndConfig.end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
	cout << "[plan] time " << plan.deltaT << endl;

	return plan;
}

/*
 * plan a motion using a tree-based planner (eg an rrt)
 */
onlineActiveModeler::motionPlanInfo onlineActiveModeler::planFreeFlyingPathWithTreeBasedPlanner(const std::vector<OpenRAVE::dReal>& planStartConfig, const std::vector<OpenRAVE::dReal>& targetConfig)
{
	motionPlanInfo plan;

	const float maxPlanningTime = 3; //in s; TODO ?
	cout << "starting ompl rrt" << endl;

	const auto checkStateValidity = [this](const omplb::State* s) -> bool
	{
		const omplb::SE3StateSpace::StateType* state = s->as<omplb::SE3StateSpace::StateType>();
		const rgbd::eigen::Affine3f camPoseWrtRaveWorld = rgbd::eigen::Translation3f(state->getX(), state->getY(), state->getZ()) * rgbd::eigen::Quaternionf(state->rotation().w, state->rotation().x, state->rotation().y, state->rotation().z);
		const rgbd::eigen::Vector3f camPosWrtRaveWorld = camPoseWrtRaveWorld * rgbd::eigen::Vector3f(0, 0, 0);
		return tsdfFreeAtPos(camPosWrtRaveWorld); //includes a check for being in map bounds
	};

	omplb::StateSpacePtr space(new omplb::SE3StateSpace());
	omplb::RealVectorBounds bounds(3); //for just the translation part of the space; ompl doesn't let us restrict the rotation part
	bounds.setLow(-3); //in m; TODO ?
	bounds.setHigh(3);
	space->as<omplb::SE3StateSpace>()->setBounds(bounds);
	omplb::SpaceInformationPtr si(new omplb::SpaceInformation(space));
	si->setStateValidityChecker(boost::bind<bool>(checkStateValidity, _1));
	omplb::RealVectorStateSpace* subspace0 = space->as<omplb::CompoundStateSpace>()->as<omplb::RealVectorStateSpace>(0);
	subspace0->setLongestValidSegmentFraction(.01 / subspace0->getMaximumExtent()); //as fraction of max extent of the space
	space->as<omplb::CompoundStateSpace>()->as<omplb::SO3StateSpace>(1)->setLongestValidSegmentFraction(.1); //really could even use 1 here; angle doesn't affect collisions
	si->setup();
	omplb::ScopedState<> startState(space), goalState(space);
	omplb::SE3StateSpace::StateType* s0 = startState->as<omplb::SE3StateSpace::StateType>(), *g0 = goalState->as<omplb::SE3StateSpace::StateType>();
	//ompl requires that the norm of each quaternion be within 1e-9 of 1, which is ok because it stores quaternions as doubles, but ours are stored as floats so we have to convert
	rgbd::eigen::Quaterniond tempq(planStartConfig[0], planStartConfig[1], planStartConfig[2], planStartConfig[3]);
	tempq.normalize();
	s0->rotation().w = tempq.w();
	s0->rotation().x = tempq.x();
	s0->rotation().y = tempq.y();
	s0->rotation().z = tempq.z();
	s0->setX(planStartConfig[4]);
	s0->setY(planStartConfig[5]);
	s0->setZ(planStartConfig[6]);
	//ompl requires that the norm of each quaternion be within 1e-9 of 1, which is ok because it stores quaternions as doubles, but ours are stored as floats so we have to convert
	tempq = rgbd::eigen::Quaterniond(targetConfig[0], targetConfig[1], targetConfig[2], targetConfig[3]);
	tempq.normalize();
	g0->rotation().w = tempq.w();
	g0->rotation().x = tempq.x();
	g0->rotation().y = tempq.y();
	g0->rotation().z = tempq.z();
	g0->setX(targetConfig[4]);
	g0->setY(targetConfig[5]);
	g0->setZ(targetConfig[6]);
#if 0 //visualization
	cout << "low: "; std::copy(space->as<omplb::SE3StateSpace>()->getBounds().low.begin(), space->as<omplb::SE3StateSpace>()->getBounds().low.end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
	cout << "high: "; std::copy(space->as<omplb::SE3StateSpace>()->getBounds().high.begin(), space->as<omplb::SE3StateSpace>()->getBounds().high.end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
	cout << "bounds checks: " << space->satisfiesBounds(s0) << " " << space->satisfiesBounds(g0) << endl;
	cout << "more checks: " << space->as<omplb::CompoundStateSpace>()->as<omplb::RealVectorStateSpace>(0)->satisfiesBounds((*g0)[0]) << ' ' << space->as<omplb::CompoundStateSpace>()->as<omplb::SO3StateSpace>(1)->satisfiesBounds((*g0)[1]) << endl;
#endif
	omplb::ProblemDefinitionPtr pdef(new omplb::ProblemDefinition(si));
	pdef->setStartAndGoalStates(startState, goalState);
	const omplb::PlannerPtr planner(new omplg::RRTConnect(si));
	planner->setProblemDefinition(pdef);
	planner->setup();
	const omplb::PlannerStatus solved = planner->solve(maxPlanningTime);
	if(solved)
	{
		const boost::shared_ptr<omplg::PathGeometric> solnptr = boost::dynamic_pointer_cast<omplg::PathGeometric>(pdef->getSolutionPath());
		const omplg::PathGeometric& soln = *solnptr;
		plan.waypoints.resize(soln.getStateCount());
		for(size_t i = 0; i < soln.getStateCount(); i++)
		{
			const omplb::SE3StateSpace::StateType* state = dynamic_cast<const omplb::SE3StateSpace::StateType*>(soln.getState(i));
			plan.waypoints[i].resize(7);
			plan.waypoints[i][0] = state->rotation().w;
			plan.waypoints[i][1] = state->rotation().x;
			plan.waypoints[i][2] = state->rotation().y;
			plan.waypoints[i][3] = state->rotation().z;
			plan.waypoints[i][4] = state->getX();
			plan.waypoints[i][5] = state->getY();
			plan.waypoints[i][6] = state->getZ();
		}
		plan.reachesTarget = true;
		plan.deltaT = 2; //what we're claiming the timing is now; the number doesn't matter, but make it slow for collision checking purposes
	}

	return plan;
}

/*
 * plan a motion for a kinematic-chain robot using a planner in openrave
 */
onlineActiveModeler::motionPlanInfo onlineActiveModeler::planKinematicChainPathWithOpenRAVE(const std::vector<OpenRAVE::dReal>& planStartConfig, const std::vector<OpenRAVE::dReal>& targetConfiguration, const OpenRAVE::EnvironmentBasePtr& planningEnv)
{
	onlineActiveModeler::motionPlanInfo result; //a default-constructed motionPlanInfo represents planning failure

	const OpenRAVE::RobotBasePtr planningRobot = planningEnv->GetRobot(robotCamHandler->getRobotName());
	robotInterface->setRAVERobotConfiguration(planningRobot, planStartConfig);
	OpenRAVE::RobotBase::ManipulatorPtr manip = planningRobot->GetActiveManipulator();
	const std::vector<int> activeDOFIndices = manip->GetArmIndices();

	rgbd::timer t;
	OpenRAVE::TrajectoryBasePtr traj = OpenRAVE::RaveCreateTrajectory(planningEnv, "");
#ifdef PLAN_WITH_CHOMP
	/*
		do motion planning with orcdchomp, an openrave plugin for CHOMP
		computedistancefield
			- kinbody: name (call once per kinbody, not incl. the robot)
		create
			- robot
			- adofgoal: final configuration (eg final configuration if will optz in configuration space, which arun says we have to if we use openrave)
			or
			- starttraj: initial trajectory (matrix w/ one config per row)
			- lambda: learning rate (arun used 200 or 500)
			- n_points: in the trajectory
			- epsilon: obstacle tolerance in m
			- obs_factor: tradeoff btwn obstacles and smoothness (arun has never edited)
		iterate() NOT in a loop
			- run: ptr to run struct
			- n_iter OR max_time in s
		gettraj
			- run: ptr as a sting
			- retime_trajectory: to enforce dof velocity limits
		destroy
			- run
		*/

	std::ostringstream outstr;
	outstr.precision(12); //increase the significant digits we'll give to and get from chomp, because why not -- TODO is this important? maybe helps w/ the openrave traj numerical check?

	outstr.str("");
{
	std::stringstream args;
	/*
	 * if it's getting too close to obstacles, can increase obs_factor or epsilon (obs_factor is multiplier in objective value; epsilon is max distance from obstacles at which obstacle cost is nonzero);
	 * if you're using t-chomp, going through obstacles can also be caused by having too much time available, so can decrease time_horizon or stop using t-chomp (since its objective function is arguably just bad)
	 *
	 * gradient step size is 1 / lambda (default lambda = 200)
	 *
	 * to use t-chomp, add 'optimize_time time_horizon <#seconds>'
	 */
//	cout << "[plan] calling create" << endl;
	args << "create robot " << robotCamHandler->getRobotName();
	//send cloned robots so can parallelize forward kinematics
	args << " robotsforFK " << "\"";
	for(size_t j = 0; j < raveEnvCopies.size(); j++)
	{
		std::vector<OpenRAVE::RobotBasePtr> robots;
		raveEnvCopies[j]->GetRobots(robots);
		ASSERT_ALWAYS(robots.size() == 1);
		args << robots[0].get() << " ";
	}
	args << "\"";
	args << " adofgoal " << "\"";
	std::copy(targetConfiguration.begin(), targetConfiguration.end(), std::ostream_iterator<OpenRAVE::dReal>(args, " "));
	args << "\"" << " lambda 300 n_points 100 epsilon .3"; //lambda 500? 200?
	try
	{
		if(!chompModule->SendCommand(outstr, args))
		{
			ASSERT_ALWAYS(false && "create failed");
		}
	}
	catch(const OpenRAVE::openrave_exception& x)
	{
		cout << "exception in chomp create: " << x.what() << endl;
		ASSERT_ALWAYS(false);
	}
}
	const std::string runptrstr = outstr.str();
	outstr.str("");
//	cout << "[plan] calling iterate" << endl;
{
	std::stringstream args;
	args << "iterate run " << runptrstr << " n_iter 150";
	if(!chompModule->SendCommand(outstr, args))
	{
		ASSERT_ALWAYS(false && "iterate failed");
	}
}
	outstr.str("");
//	cout << "[plan] calling gettraj" << endl;
{
	std::stringstream args;
	args << "gettraj run " << runptrstr << " retime_trajectory";
//#define USE_CHOMP_COLLISION_CHECKING //else use our own code so we can get details
#ifndef USE_CHOMP_COLLISION_CHECKING
	args << " no_collision_check";
#endif
	try
	{
		if(!chompModule->SendCommand(outstr, args))
		{
			ASSERT_ALWAYS(false && "gettraj failed");
		}
	}
	catch(const OpenRAVE::openrave_exception& x) //probably a collision with the map during the returned trajectory
	{
		cout << "[plan] exception in motion planning: " << x.what() << endl;
		return result;
	}
}
	//EVH: we've customized orcdchomp so that gettraj returns the trajectory even if it throws a collision exception, so we can debug
	std::istringstream instr(outstr.str());
	const std::string instrCopy = instr.str();
	try
	{
		traj->deserialize(instr);
	}
	catch(const std::exception& x) //this happens, eg, when dof limits are exceeded in the returned trajectory
	{
		cout << "[plan] couldn't parse traj from rave output; output follows" << endl;
		cout << instrCopy << endl;

		outstr.str("");
	//	cout << "[plan] calling destroy" << endl;
	{
		std::stringstream args;
		args << "destroy run " << runptrstr;
		if(!chompModule->SendCommand(outstr, args))
		{
			ASSERT_ALWAYS(false && "destroy failed");
		}
	}

		return result;
	}
	outstr.str("");
//	cout << "[plan] calling destroy" << endl;
{
	std::stringstream args;
	args << "destroy run " << runptrstr;
	if(!chompModule->SendCommand(outstr, args))
	{
		ASSERT_ALWAYS(false && "destroy failed");
	}
}
	outstr.str("");
#else //plan with openrave's default planner
	OpenRAVE::PlannerBasePtr planner = RaveCreatePlanner(planningEnv, "birrt");
	raveEnvLock envLock(planningEnv, "planningEnv in planning"); // lock environment
	planningRobot->SetActiveDOFs(manip->GetArmIndices());

	OpenRAVE::PlannerBase::PlannerParametersPtr params(new OpenRAVE::PlannerBase::PlannerParameters);
	params->_nMaxIterations = 4000; // max iterations before failure
	params->SetRobotActiveJoints(planningRobot); // set planning configuration space to current active dofs
	params->vgoalconfig.resize(planningRobot->GetActiveDOF());
	params->vgoalconfig = targetConfiguration;
	planningRobot->GetActiveDOFValues(params->vinitialconfig);
	//params->_sPostProcessingPlanner = "lineartrajectoryretimer"; //path smoother; TODO ?; for now we do smoothing/retiming ourselves
	cout << "smoother " << params->_sPostProcessingPlanner << endl;

	if(!planner->InitPlan(planningRobot, params)) planningSuccess = false;
	else
	{
		if(!planner->PlanPath(traj)) planningSuccess = false;
		else
		{
			const float vel_limit = 0.2, acc_limit = 0.2; //TODO ?; chomp uses .2, .2
			OpenRAVE::planningutils::RetimeActiveDOFTrajectory(traj, planningRobot, false/* use existing timestamps in traj */, vel_limit, acc_limit, ""/* name of planner to use (there's a default) */, ""/* extra planner params */);
		}
	}
#endif

	/*
	 * due to limitations in marvin's trajectory-running service (though there's already a trajectory class that can handle arbitrary timestamps; it just still has to be used in the python arm driver),
	 * send a resampled trajectory
	 */
	result.deltaT = .02; //something small so that the traj points we retime later will be evenly spaced to start with; TODO ?
	result.waypoints = std::move(sampleRaveTrajectory(traj, result.deltaT, true/* edit deltaT to make it divide the duration evenly */, planningRobot));
	result.reachesTarget = !result.waypoints.empty(); //using these planners, if we got a path it reaches the goal

	return result;
}

/*
 * run motion planning
 *
 * targetCamPoseInfo.viewingConfiguration, if not empty, should be a valid solution for the given camera pose
 *
 * if planning fails, return an empty list of waypoints
 */
onlineActiveModeler::motionPlanInfo onlineActiveModeler::planPath(const suggestedCamPoseInfo& targetCamPoseInfo, const std::shared_ptr<precomputedEnvCollisionData>& collisionCheckingEnvData)
{
//	const VolumeModelerAllParams& curSceneMapParams = onlineMapper->getCurSceneMapParams();
	const OpenRAVE::RobotBasePtr planningRobot = planningEnv->GetRobot(robotCamHandler->getRobotName());

	/*
	 * figure out what configuration we'll start planning from
	 */
{
	std::lock_guard<std::mutex> lock(planStartConfigMux);
	if(resetPlanStartConfig)
	{
		/*
		 * set the robot configuration we'll start motion planning from to the current configuration
		 */
		const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();
		planStartConfig = curConfig;
		resetPlanStartConfig = false;
	}
}

	motionPlanInfo plan;
	bool planningSuccess = false; //to be initialized during planning; setting here jic
	rgbd::timer t;

	/*
	 * get robot configuration for end of plan
	 */
	std::vector<OpenRAVE::dReal> targetConfiguration;
	if(targetCamPoseInfo.viewingConfiguration.empty())
	{
		if(targetCamPoseInfo.camPoseWrtMap)
		{
			const rgbd::eigen::Affine3f camPoseWrtRaveWorld = curMapPoseWrtRaveWorld * targetCamPoseInfo.camPoseWrtMap.get();
			targetConfiguration = std::move(robotInterface->getConfigurationForCamPoseWrtRaveWorld(planningRobot, robotCamHandler, camPoseWrtRaveWorld));
			if(targetConfiguration.empty()) //if we failed to find a configuration for this cam pose
				return plan;
		}
	}
	else
	{
		targetConfiguration = targetCamPoseInfo.viewingConfiguration;
	}
	cout << "[plan] planning from config: "; std::copy(planStartConfig.begin(), planStartConfig.end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
	cout << "[plan] planning to config: "; std::copy(targetConfiguration.begin(), targetConfiguration.end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
	t.stop("[plan] get planning goal config");

#ifdef DEBUG_PLANNING //visualization
{
	rgbd::timer t;
	triangulatedMesh curMapMeshEdited = *curMapMeshSeenOnly;
	const auto& tris = curMapMesh->getTriangles();
	curMapMeshEdited.allocateVertices(curMapMeshEdited.numVertices() + 3);
	curMapMeshEdited.allocateTriangles(curMapMeshEdited.numTriangles() + 1);
	for(size_t k = 0; k < 3; k++)
	{
		rgbd::pt v = targetCamPoseInfo.triVertices[k];
		v.rgb = rgbd::packRGB(255, 0, 0); //paint the target triangle red
		curMapMeshEdited.setVertex(curMapMeshEdited.numVertices() - 3 + k, v);
	}

	const rgbd::eigen::Vector3f centroid = (rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[targetCamPoseInfo.triIndex].v[0])) + rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[targetCamPoseInfo.triIndex].v[1])) + rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[targetCamPoseInfo.triIndex].v[2]))) / 3;
	triangulatedMesh viewtm = makeOctahedronMesh(centroid, .05/* dx */, rgbd::eigen::Vector3f(1, 0, 0));
	curMapMeshEdited.append(viewtm);

	curMapMeshEdited.setTriangle(curMapMeshEdited.numTriangles() - 1, triangulatedMesh::triangle{{{(uint32_t)(curMapMeshEdited.numVertices() - 3), (uint32_t)(curMapMeshEdited.numVertices() - 2), (uint32_t)(curMapMeshEdited.numVertices() - 1)}}});

	std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWithScoresForVis(2);
	camPosesWithScoresForVis[0].first = curMapPoseWrtRaveWorld.inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(planStartConfig) * robotCamHandler->getCamPoseWrtRobotBase(planStartConfig);
	camPosesWithScoresForVis[0].second = 1;
	camPosesWithScoresForVis[1].first = curMapPoseWrtRaveWorld.inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(targetConfiguration) * robotCamHandler->getCamPoseWrtRobotBase(targetConfiguration);
	camPosesWithScoresForVis[1].second = 1;
	visualizeMeshWithCameraPoses(curMapMeshEdited, camPosesWithScoresForVis, framePosesWrtCurMap[0] * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -1))/* viewingPoseWrtMap */, false/* show cur pose */,
				fs::path(), outdir / (boost::format("pathStartNGoal%1%.ply") % frameIndex).str());
	t.stop("[plan] visualize path start & goal");
}
#endif

	/*
	 * the actual planning
	 */
	t.restart();
#define PLAN_SMALL_MOTION //just move a bit toward the target pose (TODO in general this can get stuck; when/how do we want to make use of a complete planner?)
#ifdef PLAN_SMALL_MOTION
	plan = planPathByConfigInterpolation(planStartConfig, targetConfiguration);
#else
	if(robotInterface->getRobotName() == "BarrettWAM")
	{
		plan = planKinematicChainPathWithOpenRAVE(planStartConfig, targetConfiguration, planningEnv);
	}
	else if(robotInterface->getRobotName() == "FreeFlyingCamera")
	{
		plan = planFreeFlyingPathWithTreeBasedPlanner(planStartConfig, targetConfiguration, currentDeltaT);
	}
	TODO add support for wheeled robots (eg sara)
#endif
#undef PLAN_SMALL_MOTION
	planningSuccess = !plan.waypoints.empty();
	t.stop("[plan] plan");

#if defined(DEBUG_PLANNING) //visualize
	if(planningSuccess)
	{
		t.restart();

		std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWithScoresForVis(plan.waypoints.size());
		for(size_t i = 0; i < plan.waypoints.size(); i++)
		{
			const rgbd::eigen::Affine3f curPoseWrtRaveWorldDuringPlan = rgbd::eigen::Translation3f(plan.waypoints[i][4], plan.waypoints[i][5], plan.waypoints[i][6]) * rgbd::eigen::Quaternionf(plan.waypoints[i][0], plan.waypoints[i][1], plan.waypoints[i][2], plan.waypoints[i][3]);
			camPosesWithScoresForVis[i].first = curMapPoseWrtRaveWorld.inverse() * curPoseWrtRaveWorldDuringPlan;
			camPosesWithScoresForVis[i].second = (float) i / plan.waypoints.size();
		}

	{
		triangulatedMesh curMapMeshEdited = *curMapMeshSeenOnly;
		const auto& tris = curMapMesh->getTriangles();
		curMapMeshEdited.allocateVertices(curMapMeshEdited.numVertices() + 3);
		curMapMeshEdited.allocateTriangles(curMapMeshEdited.numTriangles() + 1);
		for(size_t k = 0; k < 3; k++)
		{
			rgbd::pt v = curMapMesh->v(tris[targetCamPoseInfo.triIndex].v[k]);
			v.rgb = rgbd::packRGB(255, 0, 0); //paint the target triangle red
			curMapMeshEdited.setVertex(curMapMeshEdited.numVertices() - 3 + k, v);
		}

		/*
		 * put a big red thing around the target triangle
		 */
		const rgbd::eigen::Vector3f centroid = (rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[targetCamPoseInfo.triIndex].v[0])) + rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[targetCamPoseInfo.triIndex].v[1])) + rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[targetCamPoseInfo.triIndex].v[2]))) / 3;
		triangulatedMesh viewtm = std::move(makeOctahedronMesh(centroid, .04/* dx */, rgbd::eigen::Vector3f(1, 0, 0)));
		curMapMeshEdited.append(viewtm);

		curMapMeshEdited.setTriangle(curMapMeshEdited.numTriangles() - 1, triangulatedMesh::triangle{{{(uint32_t)(curMapMeshEdited.numVertices() - 3), (uint32_t)(curMapMeshEdited.numVertices() - 2), (uint32_t)(curMapMeshEdited.numVertices() - 1)}}});
		visualizeMeshWithCameraPoses(curMapMeshEdited, camPosesWithScoresForVis, framePosesWrtCurMap[0] * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -1))/* viewingPoseWrtMap */, false/* show cur pose */,
					fs::path(), outdir / (boost::format("pathPoses%1%.ply") % frameIndex).str());
	}

		t.stop("visualize path poses");
	}
#endif

#if 0 //TODO 20140521 experiment w/ retiming before collision checking, to see whether that greatly reduces collisions (verdict: no, but it might reduce total distance traveled)
	/*
	 * retiming for safety and so we don't outpace mapping
	 */
	if(planningSuccess)
	{
		t.restart();

		/*
		 * scale timing so we don't move farther in one mapping iteration's runtime than mapping's small-movement assumption can handle
		 */
		plan.deltaT = .001; //what we're claiming the timing is now; doesn't matter -- it'll be retimed to be slower as necessary
		retimeTrajectoryForMapping(plan.waypoints, plan.deltaT);
		cout << "resampled traj to length " << plan.waypoints.size() << endl;
		cout << "deltaT = " << plan.deltaT << endl;

		/*
		 * limit the length of the trajectory we send so that as soon as we have the next one planned it can be executed
		 */
		const size_t prevTrajLength = plan.waypoints.size();
		plan.waypoints.resize(std::min(plan.waypoints.size(), (size_t)ceil(estimatedTrajPlanTime / plan.deltaT) + 1)); //limit what we send now to about estimatedTrajPlanTime
		cout << "using deltaT = " << plan.deltaT << "; shortening traj from " << prevTrajLength << " to " << plan.waypoints.size() << endl;
		plan.reachesTarget = (plan.waypoints.size() == prevTrajLength);

		t.stop("resample and/or retime traj");
	}

	/*
	 * collision checking
	 *
	 * do our own collision checking, rather than having chomp do it (if we're using chomp), so we can get debug info about the location, and possibly for increased speed
	 */
	if(planningSuccess)
	{
		t.restart();

		/*
		 * resample the trajectory at uniform spatial intervals for collision checking
		 */

		const float curPathDuration = plan.deltaT * (plan.waypoints.size() - 1);
		float pathTraversalDist = 0; //path length in meter-equivalents
		for(size_t i = 0; i < plan.waypoints.size() - 1; i++)
		{
			const rgbd::eigen::Affine3f camPoseWrtRaveWorldBefore = robotCamHandler->getRobotBasePoseWrtRaveWorld(plan.waypoints[i]) * robotCamHandler->getCamPoseWrtRobotBase(plan.waypoints[i]),
				camPoseWrtRaveWorldAfter = robotCamHandler->getRobotBasePoseWrtRaveWorld(plan.waypoints[i + 1]) * robotCamHandler->getCamPoseWrtRobotBase(plan.waypoints[i + 1]);
			float dx, da;
			const float daWeight = .5; //TODO ?
			xf::transform_difference(camPoseWrtRaveWorldBefore, camPoseWrtRaveWorldAfter, dx, da);
			pathTraversalDist += dx + daWeight * da;
		}

		OpenRAVE::dReal dtForResampling = std::min(plan.deltaT, curPathDuration * (.5 * onlineMapper->getCurSceneMapParams().volume.cell_size) / pathTraversalDist); //ensure we sample multiple times per voxel-width traversed
		const std::vector<std::vector<OpenRAVE::dReal>> trajSamples = std::move(robotInterface->sampleTrajectory(plan.waypoints, plan.deltaT, dtForResampling, true/* edit new deltaT to make it divide the duration evenly */, planningRobot));
		cout << "[plan] collision-checking " << trajSamples.size() << " traj samples" << endl;

		const std::vector<uint8_t> collisionFlags = checkRobotCollisions(collisionCheckingEnvData, trajSamples);
		for(size_t j = 0; j < collisionFlags.size(); j++)
			if(collisionFlags[j])
			{
				cout << "[plan] collision in traj: pt " << j << " of " << collisionFlags.size() << " : ["; std::copy(trajSamples[j].begin(), trajSamples[j].end(), std::ostream_iterator<double>(cout, " ")); cout << "]" << endl;
//				static int c = 0;
//				visualizeRobotPoses(curMapMesh, std::vector<std::vector<OpenRAVE::dReal>>(trajSamples.begin() + j, trajSamples.begin() + j + 1), *robotInterface, outdir / (boost::format("collTrajpt%1%_n%2%_") % frameIndex % c).str());
//				c++;
				planningSuccess = false;
				break;
			}
		t.stop("[plan] do coll check for returned traj");
	}
#else
	/*
	 * collision checking
	 *
	 * do our own collision checking, rather than having chomp do it (if we're using chomp), so we can get debug info about the location, and possibly for increased speed
	 */
	if(planningSuccess)
	{
		t.restart();

		/*
		 * resample the trajectory at uniform spatial intervals for collision checking
		 */

		const float curPathDuration = plan.deltaT * (plan.waypoints.size() - 1);
		float pathTraversalDist = 0; //path length in meter-equivalents
		for(size_t i = 0; i < plan.waypoints.size() - 1; i++)
		{
			const rgbd::eigen::Affine3f camPoseWrtRaveWorldBefore = robotCamHandler->getRobotBasePoseWrtRaveWorld(plan.waypoints[i]) * robotCamHandler->getCamPoseWrtRobotBase(plan.waypoints[i]),
				camPoseWrtRaveWorldAfter = robotCamHandler->getRobotBasePoseWrtRaveWorld(plan.waypoints[i + 1]) * robotCamHandler->getCamPoseWrtRobotBase(plan.waypoints[i + 1]);
			float dx, da;
			const float daWeight = .5; //TODO ?
			xf::transform_difference(camPoseWrtRaveWorldBefore, camPoseWrtRaveWorldAfter, dx, da);
			pathTraversalDist += dx + daWeight * da;
		}

		OpenRAVE::dReal dtForResampling = std::min(plan.deltaT, curPathDuration * (.5 * onlineMapper->getCurSceneMapParams().volume.cell_size) / pathTraversalDist); //ensure we sample multiple times per voxel-width traversed
		const std::vector<std::vector<OpenRAVE::dReal>> trajSamples = std::move(robotInterface->sampleTrajectory(plan.waypoints, plan.deltaT, dtForResampling, true/* edit new deltaT to make it divide the duration evenly */, planningRobot));
		cout << "[plan] collision-checking " << trajSamples.size() << " traj samples" << endl;

		const std::vector<uint8_t> collisionFlags = checkRobotCollisions(collisionCheckingEnvData, trajSamples);
		for(size_t j = 0; j < collisionFlags.size(); j++)
			if(collisionFlags[j])
			{
				cout << "[plan] collision in traj: pt " << j << " of " << collisionFlags.size() << " : ["; std::copy(trajSamples[j].begin(), trajSamples[j].end(), std::ostream_iterator<double>(cout, " ")); cout << "]" << endl;
//				static int c = 0;
//				visualizeRobotPoses(curMapMesh, std::vector<std::vector<OpenRAVE::dReal>>(trajSamples.begin() + j, trajSamples.begin() + j + 1), *robotInterface, outdir / (boost::format("collTrajpt%1%_n%2%_") % frameIndex % c).str());
//				c++;
				planningSuccess = false;
				break;
			}
		t.stop("[plan] do coll check for returned traj");
	}

	/*
	 * retiming for safety and so we don't outpace mapping
	 */
	if(planningSuccess)
	{
		t.restart();

		/*
		 * scale timing so we don't move farther in one mapping iteration's runtime than mapping's small-movement assumption can handle
		 */
		plan.deltaT = .001; //what we're claiming the timing is now; doesn't matter -- it'll be retimed to be slower as necessary
		retimeTrajectoryForMapping(plan.waypoints, plan.deltaT);
		cout << "resampled traj to length " << plan.waypoints.size() << endl;
		cout << "deltaT = " << plan.deltaT << endl;

		/*
		 * limit the length of the trajectory we send so that as soon as we have the next one planned it can be executed
		 */
		const size_t prevTrajLength = plan.waypoints.size();
		plan.waypoints.resize(std::min(plan.waypoints.size(), (size_t)ceil(estimatedTrajPlanTime / plan.deltaT) + 1)); //limit what we send now to about estimatedTrajPlanTime
		cout << "using deltaT = " << plan.deltaT << "; shortening traj from " << prevTrajLength << " to " << plan.waypoints.size() << endl;
		plan.reachesTarget = (plan.waypoints.size() == prevTrajLength);

		t.stop("resample and/or retime traj");

#ifdef DEBUG_PLANNING //visualize new path
	{
		rgbd::timer t;
		std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWithScoresForVis(plan.waypoints.size());
		for(size_t i = 0; i < plan.waypoints.size(); i++)
		{
			const OpenRAVE::Transform ravePose = robotSpecFreeFlyingCamera::config2raveXform(plan.waypoints[i]);
			const rgbd::eigen::Affine3f curPoseWrtRaveWorldDuringPlan = raveXform2eigenXform(ravePose);
			camPosesWithScoresForVis[i].first = curMapPoseWrtRaveWorld.inverse() * curPoseWrtRaveWorldDuringPlan;
			camPosesWithScoresForVis[i].second = (float) i / result.waypoints.size();
		}
		planningSuccess = true;
		triangulatedMesh curMapMeshEdited = *curMapMesh;
		const auto& tris = curMapMesh->getTriangles();
		curMapMeshEdited.allocateVertices(curMapMeshEdited.numVertices() + 3);
		for(size_t k = 0; k < 3; k++)
		{
			rgbd::pt v = curMapMeshEdited.v(tris[targetCamPoseInfo.triIndex].v[k]);
			v.rgb = rgbd::packRGB(255, 0, 0); //paint the target triangle red
			curMapMeshEdited.setVertex(curMapMeshEdited.numVertices() - 3 + k, v);
		}

		/*
		 * put a big red thing around the target triangle
		 */
		const rgbd::eigen::Vector3f centroid = (rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[targetCamPoseInfo.triIndex].v[0])) + rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[targetCamPoseInfo.triIndex].v[1])) + rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curMapMesh->v(tris[targetCamPoseInfo.triIndex].v[2]))) / 3;
		triangulatedMesh viewtm = std::move(makeOctahedronMesh(centroid, .04/* dx */, rgbd::eigen::Vector3f(1, 0, 0)));
		curMapMeshEdited.append(viewtm);

		curMapMeshEdited.setTriangle(targetCamPoseInfo.triIndex, triangulatedMesh::triangle{{{(uint32_t)(curMapMeshEdited.numVertices() - 3), (uint32_t)(curMapMeshEdited.numVertices() - 2), (uint32_t)(curMapMeshEdited.numVertices() - 1)}}});
		visualizeMeshWithCameraPoses(curMapMeshEdited, camPosesWithScoresForVis, framePosesWrtCurMap[0] * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -1))/* viewingPoseWrtMap */, false/* show cur pose */,
					fs::path(), outdir / (boost::format("pathPosesResampled%1%.ply") % frameIndex).str());
		t.stop("visualize resampled path");
	}
#endif
	}
#endif

	if(planningSuccess)
	{
		t.restart();
		/*
		 * visualize cur pose, cur goal pose
		 */
		std::vector<std::pair<rgbd::eigen::Affine3f, float>> camPosesWrtMapWithScoresForVis;
		const std::vector<OpenRAVE::dReal> goalConfig = trajSender->getCurGoalConfiguration();
		if(!goalConfig.empty())
		{
			camPosesWrtMapWithScoresForVis.resize(1);
			const rgbd::eigen::Affine3f goalCamPoseWrtRobotBase = robotCamHandler->getCamPoseWrtRobotBase(goalConfig);
			camPosesWrtMapWithScoresForVis[0].first = curMapPoseWrtRaveWorld.inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(goalConfig) * goalCamPoseWrtRobotBase;
			camPosesWrtMapWithScoresForVis[0].second = 1;
		}
		targetCamPoseImg = visualizeMeshWithCameraPoses(*curMapMeshSeenOnly, camPosesWrtMapWithScoresForVis, framePosesWrtCurMap[frameIndex] * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, -1))/* viewingPoseWrtMap */,
			true/* show cur pose */, "");
		t.stop("[plan] visualize cur and goal poses for gui");
	}
	else
	{
		plan.waypoints.clear(); //signifier of planning failure in the return value
	}

	return plan;
}

/*
 * visualize the traj and send it to the robot
 */
void onlineActiveModeler::enqueueTrajectory(const motionPlanInfo& plan)
{
	if(!inSimulation) //in simulation, moving the robot that's used to do simulated sensing screws everything up; TODO 20131229 we now have a separate vis env, but that's the robot we get sensing dof values from so we can't fiddle with it
	{
		/*
		 * visualize traj
		 */
		rgbd::timer t;
	{
		OpenRAVE::RobotBasePtr visrobot = visEnv->GetRobot(robotCamHandler->getRobotName());
		for(size_t q = 0; q < plan.waypoints.size(); q++)
		{

		{
			raveEnvLock envLock(visEnv, "visEnv in visualizing plan"); // lock environment
			robotInterface->setRAVERobotConfiguration(visrobot, plan.waypoints[q]);
		}
			std::this_thread::sleep_for(std::chrono::microseconds((int32_t)(plan.deltaT * .01/* .1 here means 10x real-time; TODO ? */ * 1e6)));
		}
		cout << "vis (sped up) done" << endl;
		for(int i = 0; i < 100; i++) cout << "\n";
		cout << endl;
		int qq; std::cin >> qq;
	}
		t.stop("visualize traj");
	}

	if(plan.waypoints.size() > 1)
	{
		const int64_t newPlanID = nextPlanIDToUse++;
		if(plan.reachesTarget)
		{
			plansReachingTargets[newPlanID] = curTargetID;
			ASSERT_ALWAYS(lastTargetReachedByExistingPlan < curTargetID); //debugging; TODO remove
			lastTargetReachedByExistingPlan = curTargetID;
		}
		trajSender->addTrajectory(newPlanID, 0/* traj start time, unused */, plan.waypoints, plan.deltaT);

		/*
		 * update stats
		 */
		for(size_t i = 0; i < plan.waypoints.size() - 1; i++)
		{
			const rgbd::eigen::Affine3f camPoseBefore = robotCamHandler->getRobotBasePoseWrtRaveWorld(plan.waypoints[i]) * robotCamHandler->getCamPoseWrtRobotBase(plan.waypoints[i]),
				camPoseAfter = robotCamHandler->getRobotBasePoseWrtRaveWorld(plan.waypoints[i + 1]) * robotCamHandler->getCamPoseWrtRobotBase(plan.waypoints[i + 1]);
			float dx, da;
			xf::transform_difference(camPoseBefore, camPoseAfter, dx, da);
			const float daWeight = .6; //TODO ?
			distanceTraveled += dx + daWeight * da;
		}
		cout << "[plan] total distance traveled: " << distanceTraveled << endl;

		/*
		 * set the robot dof and time values we should plan the next path segment from
		 */
		planStartConfig = plan.waypoints.back();
	}
	//TODO do we want to report a failure else?
}

/*
 * update asynchronously from modeling, which runs in a separate thread
 */
void onlineActiveModeler::updateAsync()
{
	timespec updateStartTime;
	clock_gettime(CLOCK_REALTIME, &updateStartTime);
	reportMemUsage("begin updateAsync");

	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	const size_t maxTrajsWaitingToReplan = 2/* TODO ? and use time instead of traj count? */;

	rgbd::timer t, t2;
	bool haveProcessedMaps = false;

	cout << "[plan] " << nextTargetIDToUse << " targets chosen during run" << endl;
	const std::vector<float> waitingTrajDurations = trajSender->waitingTrajDurations();
	cout << "[plan] waiting traj durations: "; std::copy(waitingTrajDurations.begin(), waitingTrajDurations.end(), std::ostream_iterator<float>(cout, " ")); cout << endl;
	cout << "[plan] last target reached " << lastTargetReached << endl;
	cout << "[plan] plans reaching targets: ";
	for(auto c : plansReachingTargets) cout << c.first << ":" << c.second << "  ";
	cout << endl;

	std::shared_ptr<precomputedEnvCollisionData> collisionCheckingEnvData; //should be set once and referenced thereafter
	/*
	 * if there's already a target pose set, try to continue moving to it
	 */
	if(curTarget)
	{
		rgbd::timer t2;

		t.restart();
		processMapsBeforeViewSelection(); //do all our reading of map info in one function
		haveProcessedMaps = true;
		t.stop("[plan] process mapping info");

		/*
		 * update id of last target pose to be reached by the camera
		 */
		rgbd::timer t;
		plansCompletedMux.lock();
		for(int64_t pid : plansCompleted)
			if(plansReachingTargets.find(pid) != plansReachingTargets.end())
				if(plansReachingTargets[pid] > lastTargetReached)
				{
					lastTargetReached = plansReachingTargets[pid];
					plansReachingTargets.erase(pid);
				}
		plansCompleted.clear();
		plansCompletedMux.unlock();
		t.stop("[plan] read plansCompleted");

		if(moveRobot && usingChomp())
		{
			/*
			 * add sdfs for all objs in the current scene to orcdchomp
			 */
			t.restart();
			computeSDFsForCHOMP();
			t.stop("compute sdfs for chomp");
		}

		cout << "[plan] have target pose" << endl;

		if(lastTargetReached >= curTargetID)
		{
			cout << "[plan] target has been reached; clearing" << endl;
			targetSwitchStateStr = (boost::format("target %d reached") % lastTargetReached).str();
			curTarget.reset();
			//TODO cancel some trajs already in the queue?
		}
		else
		{
			collisionCheckingEnvData = precomputeEnvironmentGeometryForCollisionChecking();

			const rgbd::eigen::Vector3f tgtPosInRaveWorld = curMapPoseWrtRaveWorld * curTarget->camPoseWrtMap.get() * rgbd::eigen::Vector3f(0, 0, .4/* = viewDist; TODO ? */);
			const float voxelSize = onlineMapper->getCurSceneMapParams().volume.cell_size;
			const bool targetHasBeenSeen = tsdfValueAtPos(tgtPosInRaveWorld) > -.5/* TODO ? */ * voxelSize; //= tsdfFreeAtPos(tgtPosInRaveWorld);    tsdfFree() is too conservative -- if something's really near the surface we want to stop looking at it
			if(targetHasBeenSeen)
			{
				cout << "[plan] target now in free space; clearing" << endl;
				targetSwitchStateStr = "target space became free";
				curTarget.reset();
				//TODO cancel some trajs already in the queue?
			}
			else
			{
				if(trajSender->numTrajsWaiting() < maxTrajsWaitingToReplan //don't replan until we're close to the end of our plan so far
					&& lastTargetReachedByExistingPlan < curTargetID) //don't replan to a target we've already planned to reach
				{
					//try to plan to it
					motionPlanInfo plan;
					rgbd::timer t;
					plan = std::move(planPath(*curTarget, collisionCheckingEnvData));
					t.stop("[plan] run planning");
					const bool successfulPlan = !plan.waypoints.empty();
					cout << "[plan] planning successful?: " << successfulPlan << endl;
					numSuccessfulPlans += successfulPlan;
					numPlans++;
					cout << "[plan] " << numSuccessfulPlans << " of " << numPlans << " successful" << endl;

					if(successfulPlan)
					{
						rgbd::timer t;
						enqueueTrajectory(plan);
						t.stop("[exec] call executeTrajectory");
					}
					else
					{
						cout << "[plan] unable to reach cur target; clearing target" << endl;
						targetSwitchStateStr = "unable to reach target";
						curTarget.reset(); //clear the target
						//TODO interrupt the traj sender
					}
				}
			}
		}

		if(moveRobot && usingChomp())
		{
			/*
			 * remove stored SDFs from orcdchomp
			 */
			t.restart();
			clearSDFsFromCHOMP();
			t.stop("[plan] clear chomp sdfs");
		}

		t2.stop("[plan] process existing target");
	}

	/*
	 * do full planning with new pose suggestion if necessary
	 */
	timespec curTime;
	clock_gettime(CLOCK_REALTIME, &curTime);
	const float curElapsedTime = (((long)curTime.tv_sec - modelingStartTime.tv_sec) + (curTime.tv_nsec - modelingStartTime.tv_nsec) / 1e9);
	if(!curTarget && trajSender->numTrajsWaiting() < maxTrajsWaitingToReplan) //don't replan until we're close to the end of our plan so far
	{
		cout << "[plan] elapsed time " << curElapsedTime << endl;

		if(!haveProcessedMaps)
		{
			rgbd::timer t2;
			processMapsBeforeViewSelection(); //do all our reading of map info in one function
			haveProcessedMaps = true;
			t2.stop("[plan] process mapping info");
		}

		/*
		 * choose [and execute, if on robot] movement strategies
		 */
		if(moveRobot)
		{
			if(usingChomp())
			{
				/*
				 * add sdfs for all objs in the current scene to orcdchomp
				 */
				t.restart();
				computeSDFsForCHOMP();
				t.stop("compute sdfs for chomp");
			}
		}

		std::vector<onlineActiveModeler::suggestedCamPoseInfo> suggestedNextCamPoses; //for targets, each of which might take multiple plans to reach
		t.restart();
		const std::shared_ptr<camPoseSuggestionData> suggestionData = initCamPoseSuggestion(collisionCheckingEnvData); //for efficiency, we'll suggest poses a few at a time, in preference order for attempting execution
		t.stop("[plan] init cam pose sugg");
		bool successfulPlan = false;
		bool poseSuggesterExhausted = false;
		while(!successfulPlan)
		{
			rgbd::timer u;
			suggestedNextCamPoses = suggestMoreCamPoses(suggestionData, 14/* max # suggestions at a time; TODO ? */);
			numPoseSuggestionsComputed += suggestedNextCamPoses.size();
			u.stop("[plan] get another set of poses");
			if(suggestedNextCamPoses.empty())
			{
				poseSuggesterExhausted = true;
				break;
			}

			/*
			 * filter out some possible targets using various considerations
			 */
			const auto& curSceneMapParams = onlineMapper->getCurSceneMapParams();
			std::vector<uint8_t> posesToKeep(suggestedNextCamPoses.size(), true);
			for(size_t i = 0; i < suggestedNextCamPoses.size(); i++)
			{
				const poseQuantizationIndex pqi = getPoseQuantizationIndex(curMapMesh, suggestedNextCamPoses[i], curSceneMapParams);

				/*
				 * filter by seen-too-many-times-already
				 */

				if(viewsByPose(pqi.voxel)[(uint16_t)pqi.ori] > maxViewsPerQuantizedPose) //if we've already looked at this area enough times and not been able to add it to the map
				{
					cout << "[plan] abandoning suggestion: quantized pose has already been tried " << viewsByPose(pqi.voxel)[(uint16_t)pqi.ori] << " times" << endl;
					posesToKeep[i] = false;
				}

				if(posesToKeep[i])
				{
#if 1//def DEBUGGING_20140516
					/*
					 * filter by nonobservability of tri from suggested pose
					 *
					 * TODO could also do by intersecting a ray from the suggested pose along the optical axis with the aabbtree of the scene; would that be better/faster?
					 */

					boost::multi_array<uint32_t, 2> sampleIDs(boost::extents[camParams.yRes][camParams.xRes]);
					boost::multi_array<float, 2> sampleDepths(boost::extents[camParams.yRes][camParams.xRes]);
					glContext->acquire();
				{
					const triangulatedMeshRenderer::vertexColoringFunc coloringFunc = getMeshVertexColorFromPointColor;
					std::shared_ptr<triangulatedMeshRenderer> tmpRenderer(new triangulatedMeshRenderer(*curMapMesh, coloringFunc, camParams));
					sceneRenderer->acquire();
					sceneRenderer->setRenderFunc([&tmpRenderer](const rgbd::eigen::Affine3f& camPose) {tmpRenderer->render(camPose);});
					projectSceneSamplesIntoCamera(*sceneRenderer, camParams, suggestedNextCamPoses[i].camPoseWrtMap.get(), sampleIDs, sampleDepths);
					sceneRenderer->restoreRenderFunc();
					sceneRenderer->release();
				} //ensure the triangulatedMeshRenderer releases its opengl resources while its context is active
					glContext->release();
					/*
					 * if distance at center pixel is less than something small (like 10 cm, where we know readings are invalid), abandon target;
					 * also, if distance is smaller than target and return isn't invalid, abandon target and mark not viewable
					 */
					const uint32_t cx = camParams.xRes / 2, cy = camParams.yRes / 2;
					const float minDepthNearOpticalAxis = std::min(std::min(sampleDepths[cy][cx], sampleDepths[cy][cx + 1]), std::min(sampleDepths[cy + 1][cx], sampleDepths[cy + 1][cx + 1]));
					if(minDepthNearOpticalAxis < .1/* TODO ? */)
					{
						cout << "[plan] abandoning suggestion: can't see the target surface" << endl;
						posesToKeep[i] = false;
					}
					else if(minDepthNearOpticalAxis < suggestedNextCamPoses[i].targetInfo.viewDist - .005/* wiggle room; TODO ? */) //if what we see at this pixel isn't the target
					{
						uint32_t minDepthX = cx, minDepthY = cy; //locate min-depth pixel of the four
						if(sampleDepths[cy][cx + 1] < sampleDepths[minDepthY][minDepthX]) minDepthX = cx + 1;
						if(sampleDepths[cy + 1][cx] < sampleDepths[minDepthY][minDepthX]) {minDepthX = cx; minDepthY = cy + 1;}
						if(sampleDepths[cy + 1][cx + 1] < sampleDepths[minDepthY][minDepthX]) {minDepthX = cx + 1; minDepthY = cy + 1;}

						if(sampleIDs[minDepthY][minDepthX] > 0)
						{
							cout << "[plan] abandoning suggestion and marking non-viewable: can't see the target surface" << endl;
							posesToKeep[i] = false;
							//mark not viewable, to make sure we won't try to look at it again
							viewsByPose(pqi.voxel)[(uint16_t)pqi.ori] = maxViewsPerQuantizedPose + 1;
						}
					}
#endif
				}
			}
			std::vector<onlineActiveModeler::suggestedCamPoseInfo> tmpSuggestedNextCamPoses;
			copySelected(suggestedNextCamPoses, posesToKeep, std::back_inserter(tmpSuggestedNextCamPoses));
			numPoseSuggestionsDiscarded += suggestedNextCamPoses.size() - tmpSuggestedNextCamPoses.size();
			suggestedNextCamPoses = std::move(tmpSuggestedNextCamPoses);
			cout << "[plan] have discarded " << numPoseSuggestionsDiscarded << " of " << numPoseSuggestionsComputed << " total next-pose suggestions" << endl;

			if(moveRobot)
			{
				size_t i = 0;
				motionPlanInfo plan;
				while(i < suggestedNextCamPoses.size() && !successfulPlan)
				{
					cout << "[plan] trying suggested cam pose " << i << " of " << suggestedNextCamPoses.size() << endl;

					t.restart();

					plan = std::move(planPath(suggestedNextCamPoses[i], suggestionData->collisionCheckingEnvData));
					t.stop("[plan] run planning");
					successfulPlan = !plan.waypoints.empty();
					cout << "[plan] planning successful?: " << successfulPlan << endl;
					numSuccessfulPlans += successfulPlan;
					numPlans++;
					cout << "[plan] " << numSuccessfulPlans << " of " << numPlans << " successful" << endl;
					if(successfulPlan)
					{
						if(!curTarget)
						{
							curTarget.reset(new suggestedCamPoseInfo(suggestedNextCamPoses[i]));
							curTargetID = nextTargetIDToUse++;

							lastTarget = curTarget;
						}

						const poseQuantizationIndex pqi = getPoseQuantizationIndex(curMapMesh, suggestedNextCamPoses[i], curSceneMapParams);
						viewsByPose(pqi.voxel)[(uint16_t)pqi.ori]++; //update a count of times we've looked at this area from this orientation
						if(viewsByPose(pqi.voxel)[(uint16_t)pqi.ori] > maxViewsPerQuantizedPose)
							overviewedViews.push_back(std::array<int64_t, 4>{{pqi.voxel[0], pqi.voxel[1], pqi.voxel[2], (int64_t)pqi.ori}});
						allViewsPlannedTo.push_back(std::array<int64_t, 4>{{pqi.voxel[0], pqi.voxel[1], pqi.voxel[2], (int64_t)pqi.ori}});
					}

					i++;
				}

				if(successfulPlan)
				{
					cout << "[exec] executing trajectory" << endl;
					enqueueTrajectory(plan);
				}
			}
		}

		if(moveRobot)
		{
			if(usingChomp())
			{
				/*
				 * remove stored SDFs from orcdchomp
				 */
				t.restart();
				clearSDFsFromCHOMP();
				t.stop("[plan] clear chomp sdfs");
			}
		}

		if(poseSuggesterExhausted)
		{
			cout << "[plan] pose suggester is exhausted (nothing left to see); marking experiment as done" << endl;
			experimentOver = true;
		}
	}

	t2.stop("[plan] run updateAsync");


	timespec updateEndTime;
	clock_gettime(CLOCK_REALTIME, &updateEndTime);
	totalUpdateTime += (((long)updateEndTime.tv_sec - updateStartTime.tv_sec) + (updateEndTime.tv_nsec - updateStartTime.tv_nsec) / 1e9);
	updateCount++;
	cout << "[plan] avg update time " << (totalUpdateTime / updateCount) << " (" << updateCount << " runs)" << endl;

	if(trackKnownFreeSpace)
	{
		size_t numKnownFreeVoxels = 0;
		const auto& curMapParams = onlineMapper->getCurSceneMapParams();
		ASSERT_ALWAYS(curMapParams.volume_modeler.model_type == MODEL_SINGLE_VOLUME);
		for(size_t zi = 0, l = 0; zi < (size_t)curMapParams.volume.cell_count[2]; zi++) //for each voxel
			for(size_t yi = 0; yi < (size_t)curMapParams.volume.cell_count[1]; yi++)
				for(size_t xi = 0; xi < (size_t)curMapParams.volume.cell_count[0]; xi++, l++)
					if(initialSceneFreeVoxels.voxels[zi][yi][xi] == 1 //if free in true scene
					   && (*bufferDVectors[0])[l] > .3/* TODO ? */ * curMapParams.volume.cell_size)
						numKnownFreeVoxels++;
		cout << numKnownFreeVoxels << " voxels known free" << endl;
	}

	reportMemUsage("end of updateAsync");

	if(!haveProcessedMaps)
	{
		//sleep before allowing this function to be called again, to reduce the polling rate
		std::this_thread::sleep_for(std::chrono::milliseconds(400/* TODO ? */));
	}
}

/****************************************************************************************************************************************************************
 * an interface layer to abstract away the type of robot (flying, kinematic-chain, wheeled...)
 */

std::vector<uint8_t> onlineActiveModeler::checkRobotCollisions(const std::shared_ptr<precomputedEnvCollisionData>& collisionCheckingEnvData, const std::vector<std::vector<OpenRAVE::dReal>>& configurations)
{
#if 0 //20140506: make it all use the same interface --- TODO 20140517: is this causing free-flying to run into sharp edges of unseenness?
	/*
	 * this only checks for collisions with the map mesh rather than all non-free space; at the worst this means collisions will be caught after planning when we're checking at many points along a planned route, but
	 * TODO could improve to catch when the end-of-plan pose is entirely in non-free space but not touching the map mesh (is this as simple as checking the freeness of any one point on the robot?)
	 */
	return checkRAVERobotCollisions(raveEnvCopies, robotCamHandler->getRobotName(), *robotInterface, collisionCheckingEnvData, simplifiedRobotLinkMeshes, robotLinkMeshLinkIndices, configurations);
#else
	if(robotInterface->robotIsKinematic())
	{
		/*
		 * it's enough to check for collisions with the map mesh rather than all non-free space, because the robot base is in free space, so if any part of the robot is in nonfree space there will be a mesh intersection
		 */
		return checkRAVERobotCollisions(raveEnvCopies, robotCamHandler->getRobotName(), *robotInterface, collisionCheckingEnvData, simplifiedRobotLinkMeshes, robotLinkMeshLinkIndices, configurations);
	}
	else if(robotInterface->robotIsFreeFlying())
	{
		rgbd::timer t;
#if 1 //for speed, assuming the robot has more or less zero size
		std::vector<uint8_t> collisionFlags(configurations.size(), false);
#else
		/*
		 * check for collisions with the map mesh
		 */
		std::vector<uint8_t> collisionFlags = checkRAVERobotCollisions(raveEnvCopies, robotCamHandler->getRobotName(), *robotInterface, collisionCheckingEnvData, simplifiedRobotLinkMeshes, robotLinkMeshLinkIndices, configurations);
		t.stop("check rave collisions");
#endif

		/*
		 * check for poses being outside known free space
		 */
		t.restart();
		for(size_t i = 0; i < configurations.size(); i++)
			if(!collisionFlags[i])
			{
				const rgbd::eigen::Vector3f camPosWrtRaveWorld = robotCamHandler->getRobotBasePoseWrtRaveWorld(configurations[i]) * robotCamHandler->getCamPoseWrtRobotBase(configurations[i]) * rgbd::eigen::Vector3f(0, 0, 0);
				if(tsdfValueAtPos(camPosWrtRaveWorld) < .025/* TODO ? */) collisionFlags[i] = true;
			}
		t.stop("check tsdf free");

		return collisionFlags;
	}
	else ASSERT_ALWAYS(false && "unhandled robot type");
#endif
}
