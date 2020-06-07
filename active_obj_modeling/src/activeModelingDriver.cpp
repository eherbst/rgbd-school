/*
 * activeModelingDriver: active but non-interactive object modeling for an icra14 submission
 *
 * Evan Herbst
 * 7 / 1 / 13
 */

#include <iostream>
#include <fstream>
#include <mutex>
#include <boost/program_options.hpp>
#include <boost/date_time/gregorian/gregorian_types.hpp>
#include <QtWidgets/QApplication>
#include "rgbd_util/assert.h"
#include "rgbd_util/config.h"
#include "xforms/xforms.h"
#include "rgbd_depthmaps/depthIO.h"
#include "rgbd_bag_utils/contents.h"
#include "rgbd_bag_utils/rgbdBagReader2.h"
#include "rgbd_bag_utils/rgbdFrameProviderFromBagReader.h"
#include "openni_utils/openni2FrameProvider.h"
#include "cuda_util/cudaUtils.h" //initCUDA()
#include "sm_common/renderSmallCloudNicely.h"
#include "scene_differencing/approxRGBDSensorNoiseModel.h"
#include "scene_differencing/onlineDiffingGUI.h"
#include "scene_differencing/qt_timed_class.h"
#include "peter_intel_mapping_utils/conversions.h"
#include "openrave_utils/openraveUtils.h"
#include "active_vision_common/cameraFromOpenRAVEHandler.h"
#include "active_vision_common/cameraOverROSHandler.h"
#include "active_vision_common/robotSpecMarvin.h"
#include "active_vision_common/robotSpecBaxter.h"
#include "active_vision_common/robotSpecFreeFlyingCamera.h"
#include "active_vision_common/cameraOnRobotHandlerMarvin.h"
#include "active_vision_common/cameraOnRobotHandlerBaxter.h"
#include "active_vision_common/cameraOnRobotHandlerFreeFlying.h"
#include "active_vision_common/rgbdFrameProviderFromCameraOnRobot.h"
#include "active_vision_common/openraveUtils.h"
#include "active_vision_common/simulatedScene.h"
#include "active_obj_modeling/marvinContinuousTrajSender.h"
#include "active_obj_modeling/baxterContinuousTrajSender.h"
#include "active_obj_modeling/freeFlyingCameraContinuousTrajSender.h"
#include "active_obj_modeling/onlineModeler.h"
#include "active_obj_modeling/onlineActiveModeler.h"
using std::ifstream;
using std::ofstream;
using std::cout;
using std::endl;
namespace po = boost::program_options;

/*
 * store the latest update to the data structures the gui displays, which are written by the processing thread and read by the gui thread
 */
struct guiUpdateInfo
{
	guiUpdateInfo() : version(-1)
	{}

	cv::Mat frame, diffWrtPrevSceneImg, diffWrtCurSceneImg;
	cv::Mat curMapImg;
	cv::Mat curMovedMapImg;
	cv::Mat prevMovedMapImg, prevMapImg;
	cv::Mat bkgndMapImg;
	cv::Mat targetCamPoseImg;
	cv::Mat flowDifferenceImg;
	std::vector<cv::Mat_<cv::Vec3b>> newObjModelerImgs, oldObjModelerImgs;
	std::string tgtDesc; //description of view target
	cv::Mat meshValuesImg;
	int64_t version; //so we can only make the gui update if this has changed
} guiUpdate;
boost::mutex guiUpdateMux;

/*
 * to build a bkgnd map using the robot and then look at another scene with the robot base in the same position, run something like:
 *   1) gdb --args bin/activeModelingDriver -2 ROS -U 1 -G ~/proj/rgbd-ros/rgbd/rgbd-ros-pkg/scene_matching/scene_differencing/data/gen_models/noRandom_largeSigD.yaml -o /media/sdb/scene_matching/experiments/activeModeling/robolab.mapOnly20131206.gridmap -m -a 0 -C 1 -r 0
 *   2) gdb --args bin/activeModelingDriver -v /media/sdb/scene_matching/experiments/activeModeling/robolab.mapOnly20131206.gridmap/bkgndMap -2 ROS -U 1 -G ~/proj/rgbd-ros/rgbd/rgbd-ros-pkg/scene_matching/scene_differencing/data/gen_models/noRandom_largeSigD.yaml -o /media/sdb/scene_matching/experiments/activeModeling/robolab.tmp2 -a 1 -r 1 -C 1 -j 0
 * [and move things around between the two calls]
 */
int main(int argc, char* argv[])
{
	po::options_description desc("Options");
	desc.add_options()
		//for reading the bkgnd map, one or more of these (if neither is given, we use an empty prev map)
		("scene1recpath,1", po::value<fs::path>(), "path to a bag or oni; optional")
		("scene1voxeldir,v", po::value<fs::path>(), "path to a dir with a saved peter-intel map; optional (we build the map if none is provided)")

		//for reading the cur-scene recording
		("scene2recpath,2", po::value<fs::path>(), "path to a bag or oni; if == 'ONI', will stream with openni2; if == 'ROS', will stream with ros; if == 'SIM', will simulate sensor readings and arm movement")

		("cur-map-pose-wrt-prev,P", po::value<std::string>()->default_value(""), "if given, will use for interscene alignment")

		("new-map-grid,U", po::value<bool>()->default_value(false), "for peter-intel mapping of new scene: do we use a grid map (as opposed to single volume)?")

		//one of the below
		("gen-model-dumpfile,g", po::value<fs::path>(), "filepath for saved sensor noise model; use if you want a full sensor model")
		("gen-model-paramfile,G", po::value<fs::path>(), "filepath for sensor noise model params; use if you want an approx sensor model")

		//experiment setup
		("make-map-only,m", po::value<bool>()->default_value(false)->zero_tokens(), "if true, this run is just to make a bkgnd map of a small part of the scene; stop and dump a map after just a few frames")
		("do-obj-modeling,j", po::value<bool>()->default_value(true), "whether we run online object modeling")
		("active-modeling,a", po::value<bool>(), "whether we're doing active or passive online modeling (if active, can do view selection without doing the resulting movement by using -r 0)")
		("robot-name,R", po::value<std::string>()->default_value("BarrettWAM"), "openrave model name: BarrettWAM | Baxter | FreeFlyingCamera")
		("cam-on-robot,C", po::value<bool>(), "whether the camera is on the robot; used if streaming frames (default: yes iff doing active modeling)")
		("move-robot,r", po::value<bool>(), "whether to move the (real or simulated) arm to selected next views (default: yes iff cam on robot; not applicable if not doing active modeling)")
		("tri-value-func,f", po::value<std::string>()->default_value("distance_to_cam"), "valuation function for triangles as view targets in active mapping")

		("outdir,o", po::value<fs::path>(), "will be created if nec")
		;
	po::variables_map vars;
	po::store(po::command_line_parser(argc, argv).options(desc).run(), vars);
	po::notify(vars);

	fs::path noiseModelFilepath, noiseModelParamsFilepath;
	if(vars.count("gen-model-dumpfile")) noiseModelFilepath = vars["gen-model-dumpfile"].as<fs::path>();
	else
	{
		ASSERT_ALWAYS(vars.count("gen-model-paramfile"));
		noiseModelParamsFilepath = vars["gen-model-paramfile"].as<fs::path>();
	}
	const bool useGridMapsForCurScene = vars["new-map-grid"].as<bool>();
	const fs::path outdir(vars["outdir"].as<fs::path>());
	fs::create_directories(outdir);

	const std::string robotName = vars["robot-name"].as<std::string>();
	const bool makeBkgndMapOnly = vars["make-map-only"].as<bool>();
	const std::string interscenePoseStr = vars["cur-map-pose-wrt-prev"].as<std::string>();
	const bool doObjModeling = vars["do-obj-modeling"].as<bool>();
	const bool activeModeling = vars["active-modeling"].as<bool>(); //whether to do view selection
	const bool camOnRobot = vars.count("cam-on-robot") ? vars["cam-on-robot"].as<bool>() : activeModeling; //is/was the camera attached to the robot during recording of the new scene?
	const fs::path bagFilepath = vars["scene2recpath"].as<fs::path>();
	const bool useOpenni2 = (bagFilepath.string() == "ONI"), useROS = (bagFilepath.string() == "ROS"),
		runInSimulation = (bagFilepath.string() == "SIM"); //use openrave to simulate both moving the robot and getting camera frames
	const bool streamingInput = useOpenni2 || useROS; //are we getting streamed video?
	const bool moveRobot = !activeModeling ? false : (vars.count("move-robot") ? vars["move-robot"].as<bool>() : camOnRobot);
	const std::string triValueFuncStr = vars["tri-value-func"].as<std::string>();
	ASSERT_ALWAYS(!(activeModeling && makeBkgndMapOnly)); //mapping-only mode is not compatible with active stuff
	ASSERT_ALWAYS(!activeModeling || camOnRobot); //onlineActiveModeler requires cam on robot
	ASSERT_ALWAYS(!(runInSimulation && streamingInput));
	ASSERT_ALWAYS(!runInSimulation || camOnRobot); //what would it mean to simulate readings from a camera without controlling its motion?
	//TODO print all these to screen?

	rgbd::cameraSetup cams;
	cams.cam = rgbd::KINECT_640_DEFAULT; //use the same cam you used to build the scene-1 map -- get it wrong and alignment will suck; TODO parameterize
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);

	//to be shared by pretty much everything
	std::shared_ptr<openglContext> glContext;
	std::shared_ptr<viewScoringRenderer> sceneRenderer;
	glContext.reset(new openglContext(camParams.xRes, camParams.yRes));
	sceneRenderer.reset(new viewScoringRenderer(glContext, camParams, true/* use large texture buffer */));

	glContext->acquire();
	cuda::initCUDA(true/* need opengl interop */);
	glContext->release();

	ros::init(argc, argv, "activeModelingDriver");
	ros::Time::init(); //so we can use Time::now()

	configOptions cfg;

	std::shared_ptr<robotSpec> robotInterface;
	if(camOnRobot)
	{
		if(robotName == "BarrettWAM") robotInterface.reset(new robotSpecMarvin);
		else if(robotName == "Baxter") robotInterface.reset(new robotSpecBaxter);
		else if(robotName == "FreeFlyingCamera") robotInterface.reset(new robotSpecFreeFlyingCamera);
		else ASSERT_ALWAYS(false && "unknown robot");

		//for various things
		cfg.set("robotName", robotName);
		cfg.set("pathtoRobotXML", robotInterface->getRelativeRaveModelFilepath());
		cfg.set("activeManipulatorName", robotInterface->getDefaultManipulatorName());
		cfg.set("camAttachmentLinkName", robotInterface->getCamAttachmentLinkName());
		cfg.set("simulateRobot", runInSimulation);

		//for openraveUtils
		cfg.set("raveBasePath", "/home/eherbst/proj/ros-extras/openrave_planning/openrave/share/openrave-latest");

		//for cameraOnRobotHandler: if we're dumping a map, also dump the video including joints
		if(makeBkgndMapOnly)
			cfg.set("dumpRGBDJOutpath", outdir / "bkgndMap" / "rgbdjVideo.bag");
	}
	if(activeModeling)
	{
		//for onlineActiveModeler
		cfg.set("outdir", outdir);
		cfg.set("runOnRobot", moveRobot);

		cfg.set("triValueFunc", triValueFuncStr);
	}

	OpenRAVE::RaveInitialize(true);
	OpenRAVE::RaveSetDebugLevel(OpenRAVE::Level_Debug);
	cfg.set("initRave", false);
	cfg.set("initEnv", false);
	OpenRAVE::EnvironmentBasePtr visEnv;
	if(camOnRobot)
	{
		visEnv = OpenRAVE::RaveCreateEnvironment();
		robotInterface->loadRobot(visEnv, cfg);

		/*
		 * speed up adding large meshes to openrave, which we do repeatedly: most of the time for adding is in adding to the ode (default) collision checker
		 */
		OpenRAVE::CollisionCheckerBasePtr collChecker = OpenRAVE::RaveCreateCollisionChecker(visEnv, "GenericCollisionChecker");
		ASSERT_ALWAYS(collChecker);
		visEnv->SetCollisionChecker(collChecker);
	}

	std::shared_ptr<QApplication> app;
	OpenRAVE::ViewerBasePtr viewer;
	if(camOnRobot)
	{
		//this will create a qt application
		viewer = OpenRAVE::RaveCreateViewer(visEnv, "qtcoin");
		ASSERT_ALWAYS(!!viewer);
		visEnv->AddViewer(viewer);
	}
	else
	{
		app.reset(new QApplication(argc, argv));
	}

	std::shared_ptr<onlineDiffingGUI> gui(new onlineDiffingGUI(camParams));
	enum guiPane
	{
		GUI_FRAME_IMG,
		GUI_DIFF_PREV_IMG,
		GUI_DIFF_CUR_IMG,
		GUI_PREV_MAP_IMG,
		GUI_CUR_MAP_IMG,
		GUI_BKGND_MAP_IMG,
		GUI_TGT_CAM_POSE_IMG,
		GUI_NEW_OBJ_MODELER_IMGS,
		GUI_OLD_OBJ_MODELER_IMGS,
		GUI_FLOW_DIFF_IMG,
		GUI_TARGET_DESC,
		GUI_MESH_VALUES
	};
	gui->setPaneName(GUI_FRAME_IMG, "frame");
	gui->setPaneName(GUI_DIFF_PREV_IMG, "diff wrt prev");
	gui->setPaneName(GUI_DIFF_CUR_IMG, "diff wrt cur");
	gui->setPaneName(GUI_PREV_MAP_IMG, "prev map");
	gui->setPaneName(GUI_CUR_MAP_IMG, "cur map");
	gui->setPaneName(GUI_BKGND_MAP_IMG, "bkgnd map");
	gui->setPaneName(GUI_TGT_CAM_POSE_IMG, "cam pose target");
	gui->setPaneName(GUI_NEW_OBJ_MODELER_IMGS, "new obj modeler imgs");
	gui->setPaneName(GUI_OLD_OBJ_MODELER_IMGS, "old obj modeler imgs");
	gui->setPaneName(GUI_FLOW_DIFF_IMG, "flow difference");
	gui->setPaneName(GUI_TARGET_DESC, "tgt description");
	gui->setPaneName(GUI_MESH_VALUES, "mesh values");
	//call a function (here, to poll the gui update info) every so often -- if I had qt5 I'd use its nice c++11-enabled QTimer instead
	int64_t latestLoadedVersion = -1;
	timed_class t([&]()
		{
			boost::lock_guard<boost::mutex> lock(guiUpdateMux);
			if(guiUpdate.version > latestLoadedVersion)
			{
				if(guiUpdate.frame.rows > 0) gui->setImg(GUI_FRAME_IMG, guiUpdate.frame);
				if(guiUpdate.diffWrtPrevSceneImg.rows > 0) gui->setImg(GUI_DIFF_PREV_IMG, guiUpdate.diffWrtPrevSceneImg);
				if(guiUpdate.diffWrtCurSceneImg.rows > 0) gui->setImg(GUI_DIFF_CUR_IMG, guiUpdate.diffWrtCurSceneImg);
				if(guiUpdate.prevMapImg.rows > 0) gui->setImg(GUI_PREV_MAP_IMG, guiUpdate.prevMapImg);
				if(guiUpdate.curMapImg.rows > 0) gui->setImg(GUI_CUR_MAP_IMG, guiUpdate.curMapImg);
				if(guiUpdate.bkgndMapImg.rows > 0) gui->setImg(GUI_BKGND_MAP_IMG, guiUpdate.bkgndMapImg);
				if(guiUpdate.newObjModelerImgs.size() > 0) gui->setImgSet(GUI_NEW_OBJ_MODELER_IMGS, guiUpdate.newObjModelerImgs);
				if(guiUpdate.oldObjModelerImgs.size() > 0) gui->setImgSet(GUI_OLD_OBJ_MODELER_IMGS, guiUpdate.oldObjModelerImgs);
				if(guiUpdate.targetCamPoseImg.rows > 0) gui->setImg(GUI_TGT_CAM_POSE_IMG, guiUpdate.targetCamPoseImg);
				if(guiUpdate.flowDifferenceImg.rows > 0) gui->setImg(GUI_FLOW_DIFF_IMG, guiUpdate.flowDifferenceImg);
				if(guiUpdate.tgtDesc != "") gui->setText(GUI_TARGET_DESC, guiUpdate.tgtDesc);
				if(guiUpdate.meshValuesImg.rows > 0) gui->setImg(GUI_MESH_VALUES, guiUpdate.meshValuesImg);
				latestLoadedVersion = guiUpdate.version;
			}
		});

	std::shared_ptr<rgbdSensorNoiseModel> noiseModel;
	if(noiseModelFilepath.empty()) noiseModel.reset(new approxRGBDSensorNoiseModel(noiseModelParamsFilepath));
	else noiseModel.reset(new jairRGBDSensorNoiseModel(noiseModelFilepath));

	/*
	 * initialize cam handler
	 */
	std::shared_ptr<rgbdCameraHandler> rgbdCamHandler; //will always be initialized
	OpenRAVE::EnvironmentBasePtr simulationEnv; //used if simulating robot sensors & movement
	simulatedScene simScene; //simulated scene, maybe not used
	if(runInSimulation)
	{
		/*
		 * create an environment and a scene to sense
		 *
		 * we can only create one viewer, but the camera sensor requires a viewer for rendering, so the sensors need to be in the env used by that viewer;
		 * ideally we'd have them in a different one so we could see the map mesh and the simulated environment separately
		 *
		 * 20131221 no longer using the camera sensor to acquire simulated readings, so can now use separate environments if we want
		 */
#if 1
		simulationEnv = OpenRAVE::RaveCreateEnvironment();
		simScene.addToEnv(simulationEnv);
		robotInterface->loadRobot(simulationEnv, cfg);
#else
		simulationEnv = visEnv;
#endif
		//TODO need to lock visEnv?
		OpenRAVE::RobotBasePtr movingRobot = visEnv->GetRobot(robotName); //the robot used for simulating movement
		OpenRAVE::RobotBasePtr sensingRobot = simulationEnv->GetRobot(robotName); //the robot used for simulating sensing
		OpenRAVE::RobotBase::ManipulatorPtr manip = sensingRobot->GetActiveManipulator();
#if 0 //don't need if we're not using openrave for simulating sensing
		OpenRAVE::SensorBasePtr colorCamSensor, depthCamSensor;
#endif

		std::shared_ptr<triangulatedMesh> simulatedSceneMesh; //TODO set mesh in cases when that will be useful to rgbdCameraFromOpenRAVEHandler
	{
		raveEnvLock envLock(simulationEnv, "simulationEnv in driver"); // lock environment

		const std::vector<OpenRAVE::dReal> initialConfig = robotInterface->getStartConfigForSimulatedScene(simulationEnv);
		robotInterface->setRAVERobotConfiguration(movingRobot, initialConfig);

#if 0 //don't need if we're not using openrave for simulating sensing
		/*
		 * init simulated sensors (requires env lock)
		 */
		const std::vector<OpenRAVE::RobotBase::AttachedSensorPtr> sensors = sensingRobot->GetAttachedSensors();
		colorCamSensor = sensors[0]->GetSensor();
		depthCamSensor = sensors[1]->GetSensor();
		colorCamSensor->Configure(OpenRAVE::SensorBase::CC_PowerOn); //necessary for camera sensors to simulate readings
		depthCamSensor->Configure(OpenRAVE::SensorBase::CC_PowerOn); //necessary for laser sensors to simulate readings
#endif
	}

		rgbdCamHandler.reset(new rgbdCameraFromOpenRAVEHandler(camParams, robotInterface, movingRobot, sensingRobot, glContext, sceneRenderer, simulatedSceneMesh));
	}
	else if(camOnRobot)
	{
		//TODO figure out how to get compressed imgs with timestamps that match (20131009 if you just use the /compressed topic here, they don't) so we can use a time synchronizer     <-- is it because I didn't have the subscriber take CompressedImage?
		if(robotName == "BarrettWAM") //assume marvin
		{
			cfg.set("imgTopic", "/camera/rgb/image_color_sync");
			cfg.set("depthTopic", "/camera/depth/image_sync");
		}
		else if(robotName == "Baxter")
		{
			//TODO make low-rate sync work as it does for marvin
			cfg.set("imgTopic", "/camera/rgb/image_color");
			cfg.set("depthTopic", "/camera/depth_registered/image");
		}
		else throw std::runtime_error("unknown robot");
		cfg.set("spinInSeparateThread", true);

		rgbdCamHandler.reset(new cameraOverROSHandler(cfg));
	}

	/*
	 * initialize cam-on-robot interface
	 */
	std::shared_ptr<cameraOnRobotHandler> robotCamHandler; //will be valid if camOnRobot
	if(camOnRobot)
	{
		if(robotName == "BarrettWAM") //assume marvin
			robotCamHandler.reset(new cameraOnRobotHandlerMarvin(cfg, visEnv, rgbdCamHandler, runInSimulation, robotInterface));
		else if(robotName == "Baxter")
			robotCamHandler.reset(new cameraOnRobotHandlerBaxter(cfg, visEnv, rgbdCamHandler, runInSimulation, robotInterface));
		else if(robotName == "FreeFlyingCamera")
		{
			ASSERT_ALWAYS(runInSimulation);
			robotCamHandler.reset(new cameraOnRobotHandlerFreeFlying(cfg, visEnv, rgbdCamHandler, robotInterface));
		}
		else ASSERT_ALWAYS(false && "unhandled robot");
		robotCamHandler->init();
		cout << "created cam-on-robot handler" << endl;
	}

	/*
	 * set up modeler
	 */
	onlineSceneDifferencer::params differParams;
	if(vars.count("scene1recpath") || vars.count("scene1voxeldir"))
	{
		differParams.useEmptyScene1 = false;
		if(vars.count("scene1recpath")) differParams.scene1recordingPath = vars["scene1recpath"].as<fs::path>();
		if(vars.count("scene1voxeldir")) differParams.scene1voxelMapDir = vars["scene1voxeldir"].as<fs::path>();
	}
	else
	{
		differParams.useEmptyScene1 = true;
		cout << "using an empty background map" << endl;
	}
	differParams.cams = cams;
	differParams.useGridMapsForCurScene = useGridMapsForCurScene;
#if 1 //make sure we can fit the room's walls inside a single volume (grid maps are bad for this purpose since they have broken rendering); used, eg, for the simenv diffing-based experiment in the thesis
	differParams.voxelSize = .025;
#else //a good default value
	differParams.voxelSize = .01;
#endif
	boost::optional<rgbd::eigen::Affine3f> curMapPoseWrtRaveWorld;
	if(robotCamHandler)
	{
		if(!useGridMapsForCurScene)
		{
			/*
			 * center the map at the robot base
			 *
			 * if we're not running on a robot, the "robot base" will be at an arbitrary pose but will be consistent over different transforms, so we can pretend it exists
			 */
#if 1 //for close-up-scene free-flying experiment 20140306
			differParams.voxelCount = rgbd::eigen::Vector3i(256, 256, 256);
#else
			differParams.voxelCount = rgbd::eigen::Vector3i(256, 256, 256); //big enough that marvin can't reach the edges from the center
#endif
			curMapPoseWrtRaveWorld = robotCamHandler->getRobotBasePoseWrtRaveWorld(robotCamHandler->getLatestConfiguration()) * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(-differParams.voxelCount.cast<float>() * differParams.voxelSize / 2));

#if 1 //for close-up-scene free-flying experiment 20140306
			curMapPoseWrtRaveWorld = curMapPoseWrtRaveWorld.get() * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, .5));
#endif
		}
		else
		{
			curMapPoseWrtRaveWorld = robotCamHandler->getRobotBasePoseWrtRaveWorld(robotCamHandler->getLatestConfiguration());
		}

		/*
		 * get joint angles at start of new video if possible
		 */
		if(runInSimulation || streamingInput)
		{
			cout << "getting cam pose wrt base from robot" << endl;
			ASSERT_ALWAYS(robotCamHandler);
			const auto camPoseWrtBase = robotCamHandler->getCamPoseWrtRobotBase(robotCamHandler->getLatestConfiguration());
			cout << "cam pose wrt base: " << endl << camPoseWrtBase.matrix() << endl;
			differParams.frame0camPoseWrtCurMapToUse = curMapPoseWrtRaveWorld.get().inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(robotCamHandler->getLatestConfiguration()) * camPoseWrtBase;
		}
		else if(!streamingInput && bagFilepath.extension() == ".bag")
		{
			/*
			 * open bag and check for cam-pose-wrt-robot info
			 */
			boost::optional<rgbd::eigen::Transform3f> initialCamPoseWrtRobotBase;
		{
			rosbag::Bag bag(bagFilepath.string(), rosbag::bagmode::Read);
			rosbag::View view(bag, rosbag::View::TrueQuery());
			bool timeSeen = false;
			ros::Time startTime;
			for(rosbag::View::iterator i = view.begin(); i != view.end(); i++)
			{
				rosbag::MessageInstance& m = *i;
				if(!timeSeen)
				{
					startTime = m.getTime();
					timeSeen = true;
				}
				else if(m.getTime() - startTime > ros::Duration(1/* second */)) break; //stop this far into the bag

				geometry_msgs::TransformPtr j;
				if(j = m.instantiate<geometry_msgs::Transform>())
					initialCamPoseWrtRobotBase = xf::geomsg2eigen(*j);
			}
		}
			if(initialCamPoseWrtRobotBase) //if the bag contains joint angles
			{
				cout << "taking cur-map start pose from bag" << endl;
				differParams.frame0camPoseWrtCurMapToUse = curMapPoseWrtRaveWorld.get().inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(robotCamHandler->getLatestConfiguration()) * initialCamPoseWrtRobotBase.get() * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(0, 0, .1))/* TODO figure out what's really wrong; why is this needed? */;
			}
		}
	}
	else //no robot
	{
#if 0 //for objsmove2, the darmstadt back-and-forth experiment
			differParams.voxelCount = rgbd::eigen::Vector3i(512, 192, 256);
#elif 0 //for 20140205 diffing model comparison on simenv1 for thesis
			differParams.voxelCount = rgbd::eigen::Vector3i(384, 256, 512);
#else
			differParams.voxelCount = rgbd::eigen::Vector3i(256, 256, 256);
#endif
	}
	/*
	 * allow the user to provide a transform between old and new maps
	 */
	if(!interscenePoseStr.empty())
	{
#if 0 //user provides alignment btwn frame 0 of prev and cur maps
		const rgbd::eigen::Affine3f curFrame0PoseWrtOldFrame0 = xf::parseTransformString(interscenePoseStr); //we might get this from manualPtCloudAlignmentGUI, eg
		ASSERT_ALWAYS(fs::exists(differParams.scene1voxelMapDir));
		const std::vector<rgbd::eigen::Affine3f> camPoses = xf::readTransformsTextFileNoTimestamps(differParams.scene1voxelMapDir / "mapOffsetXform.dat");
		const rgbd::eigen::Affine3f prevFrame0PoseWrtPrevMap = camPoses[0];
		differParams.curMapPoseWrtOldMap = prevFrame0PoseWrtPrevMap * curFrame0PoseWrtOldFrame0 * differParams.frame0camPoseWrtCurMapToUse.get().inverse();
#else //user provides alignment btwn volumetric maps
		const rgbd::eigen::Affine3f curMapPoseWrtOldMap = xf::parseTransformString(interscenePoseStr);
		differParams.curMapPoseWrtOldMap = curMapPoseWrtOldMap;
#endif
	}
	else if(!differParams.scene1voxelMapDir.empty() && curMapPoseWrtRaveWorld)
	{
		/*
		 * if we know poses of old and new maps wrt world, compute interscene pose (assuming the rave world is at the same pose for both maps)
		 */
		const fs::path mapPosePath = differParams.scene1voxelMapDir / "mapPoseWrtRaveWorld.dat";
		if(fs::exists(mapPosePath))
		{
			ifstream infile(mapPosePath.string());
			ASSERT_ALWAYS(infile);
			std::string line;
			getline(infile, line);
			const rgbd::eigen::Affine3f prevMapPoseWrtRaveWorld = xf::parseTransformString(line);
			differParams.curMapPoseWrtOldMap = prevMapPoseWrtRaveWorld.inverse() * curMapPoseWrtRaveWorld.get();
		}
	}
	differParams.useDiffingOptz = false;
	differParams.outputDir = outdir;
	differParams.pauseAfterEachFrame = false;

	/*
	 * create modeler and related things
	 */
	std::shared_ptr<onlineObjModeler> modeler;
	std::shared_ptr<continuousTrajSender> trajSender;
	std::shared_ptr<onlineActiveModeler> viewPlanner;
	modeler.reset(new onlineObjModeler(noiseModel, differParams, doObjModeling));
	if(activeModeling)
	{
		const OpenRAVE::EnvironmentBasePtr trajSimEnv = runInSimulation ? visEnv : OpenRAVE::EnvironmentBasePtr();
		if(robotName == "BarrettWAM") //assume marvin
			trajSender.reset(new marvinContinuousTrajSender(robotCamHandler, trajSimEnv));
		else if(robotName == "Baxter")
			trajSender.reset(new baxterContinuousTrajSender(robotCamHandler, trajSimEnv));
		else if(robotName == "FreeFlyingCamera")
			trajSender.reset(new freeFlyingCameraContinuousTrajSender(robotCamHandler, robotInterface, trajSimEnv));
		else ASSERT_ALWAYS(false && "unknown robot");
		viewPlanner.reset(new onlineActiveModeler(glContext, sceneRenderer, robotInterface, robotCamHandler, trajSender, modeler, cfg, visEnv, curMapPoseWrtRaveWorld.get()));
	}

	/*
	 * configure input source
	 */
	std::shared_ptr<rgbdFrameProvider> frameProvider;
	std::shared_ptr<rgbd::rgbdBagReader2> bagReader; //not nec used
	if(runInSimulation)
	{
		ASSERT_ALWAYS(robotCamHandler); //not sure whether we ever use a simulated non-robot cam
		std::shared_ptr<cameraOnRobotRGBDFrameProvider> robotFrameProvider(new cameraOnRobotRGBDFrameProvider(robotCamHandler));
		frameProvider = robotFrameProvider;

		/*
		 * allow us to use low-texture simulated environments without mapping alignment screwing up: provide correct poses to the modeler in lieu of using mapping's alignment
		 *
		 * give the frame provider a callback so it can synchronize announcements of new frames with the corresponding camera poses
		 */

		std::shared_ptr<std::mutex> latestConfigMux(new std::mutex); //these are shared_ptrs so that the closures below will, when taking copies by value, actually have references
		std::shared_ptr<std::vector<OpenRAVE::dReal>> latestConfig(new std::vector<OpenRAVE::dReal>);
		//to be called each time a frame is requested from the frame provider, so we have a correctly synchronized camera pose for the frame, which is important when we're taking that pose as ground truth
		const auto updateCurConfigFunc = [latestConfig,latestConfigMux](const std::vector<OpenRAVE::dReal>& config)
			{
				latestConfigMux->lock();
				*latestConfig = config;
				latestConfigMux->unlock();
			};
		const auto getCurCamPoseWrtCurMapFunc = [=]()
			{
				latestConfigMux->lock();
				const std::vector<OpenRAVE::dReal> curConfig = *latestConfig;//robotCamHandler->getLatestConfiguration(); //calling the robotCamHandler here means sometimes the pose we get isn't synchronized with the latest rgbd frame
				latestConfigMux->unlock();
				return curMapPoseWrtRaveWorld.get().inverse() * robotCamHandler->getRobotBasePoseWrtRaveWorld(curConfig) * robotCamHandler->getCamPoseWrtRobotBase(curConfig);
			};
		modeler->setRunAlignment(false);
		modeler->setCurCamPoseWrtMapGetter(getCurCamPoseWrtCurMapFunc);
		robotFrameProvider->addConfigUpdateCallback(updateCurConfigFunc);
	}
	else if(useOpenni2) //no robot involved
	{
		FrameProviderOpenni2Params readerParams;
		frameProvider.reset(new openni2FrameProvider(readerParams));
	}
	else if(useROS)
	{
		frameProvider.reset(new cameraOnRobotRGBDFrameProvider(robotCamHandler));
	}
	else
	{
		ASSERT_ALWAYS(fs::exists(bagFilepath));
		bagReader.reset(new rgbd::rgbdBagReader2(bagFilepath, 0, 1000000, 0/* 0 usually; 1 for icra14 video *//* frameskip */, 0/* num prev frames to keep */));
		frameProvider.reset(new archiveReaderRGBDFrameProvider(*bagReader));
	}

	std::thread nonguiThread([&]()
		{
	/************************************************************************************************************************************************************************************************
	 * main loop
	 *
	 * start planning in a separate thread from modeling
	 * (planning takes much longer per frame than modeling and we don't want it to break modeling's small-motion assumption)
	 */
	std::shared_ptr<std::thread> planningThread;
	cout << "starting main loop" << endl;
	size_t frameIndex = 0;
	bool experimentOver = false;
	rgbd::timer t2;
	while(!experimentOver)
	{
		rgbd::timer t;
		/****************************************************************
		 * read and process
		 */

		rgbd::timer u;
		rgbdFrame frame(camParams);
		boost::posix_time::ptime frameTime;
		if(!frameProvider->getNextFrame(frame.getColorImgRef(), frame.getDepthImgRef(), frameTime)) break;
		u.stop("get frame");
#if 0 //dump frames to disk
		cv::imwrite((outdir / (boost::format("rgb%1%.png") % frameIndex).str()).string(), frame.getColorImg());
		cv::imwrite((outdir / (boost::format("depth%1%.png") % frameIndex).str()).string(), frame.getDepthImg());
#endif
	{
		u.restart();
		if(!modeler->nextFrame(frame, frameTime))
		{
			cout << "*** alignment failed" << endl;
			ASSERT_ALWAYS(false);
		}
		u.stop("call nextFrame");
		if(viewPlanner)
		{
			rgbd::timer t;
			viewPlanner->updatePerModelingFrame(frame, frameTime);
			t.stop("call updatePerFrame");
			t.restart();
			if(viewPlanner->isExperimentOver()) experimentOver = true;
			t.stop("call isExperimentOver");
		}
	}

		/*
		 * ensure at least one frame is processed by modeling before we initiate motion planning, which runs concurrently with this loop
		 */
		if(frameIndex == 2 && viewPlanner)
		{
			planningThread.reset(new std::thread([&]()
				{
					while(!experimentOver)
					{
						viewPlanner->updateAsync();
					}
				}));
		}

#if 1
		/****************************************************************
		 * visualize
		 *
		 * this runs in the same thread as the online mapping, so we don't need synchronization with the mapper
		 */
		u.restart();
		guiUpdate.frame = frame.getColorImg();
		guiUpdate.version = frameIndex;
	{
		boost::lock_guard<boost::mutex> lock(guiUpdateMux);
	{
		guiUpdate.diffWrtPrevSceneImg = modeler->getDiffingWrtOldSceneVisImg();
		guiUpdate.diffWrtCurSceneImg = modeler->getDiffingWrtNewSceneVisImg();
		guiUpdate.prevMapImg = modeler->getPrevMapImg();
		guiUpdate.curMapImg = modeler->getCurMapImg();
		guiUpdate.bkgndMapImg = modeler->getBkgndMapImg();
		guiUpdate.newObjModelerImgs = modeler->getCloseUpNewModelImgs();
		guiUpdate.oldObjModelerImgs = modeler->getCloseUpOldModelImgs();
		if(activeModeling)
		{
			//TODO prevent data races with the view planner here
			guiUpdate.targetCamPoseImg = viewPlanner->getTargetCamPoseImg();
			guiUpdate.tgtDesc = viewPlanner->getTargetCamPoseDescription() + "\n\n" + viewPlanner->getTargetSwitchInfo();
			guiUpdate.meshValuesImg = viewPlanner->getMeshValuesImg();
		}

#if 0 //visualization 20140204
		std::vector<cv::Mat_<cv::Vec3b>> newModelImgs2 = modeler->getNewModelImgsV3();
		for(size_t i = 0; i < newModelImgs2.size(); i++) cv::imwrite((outdir / (boost::format("newObjV3_%1%_%2%.png") % frameIndex % i).str()).string(), newModelImgs2[i]);
		std::vector<cv::Mat_<cv::Vec3b>> oldModelImgs2 = modeler->getOldModelImgsV3();
		for(size_t i = 0; i < oldModelImgs2.size(); i++) cv::imwrite((outdir / (boost::format("oldObjV3_%1%_%2%.png") % frameIndex % i).str()).string(), oldModelImgs2[i]);
#endif
	}

#if 1
		cv::imwrite((outdir / (boost::format("frame%1$08d.png") % frameIndex).str()).string(), frame.getColorImg());
		rgbd::writeDepthMapValuesImg(frame.getDepthImg(), outdir / (boost::format("depth%1$08d.png") % frameIndex).str());
		cv::imwrite((outdir / (boost::format("bkgndMap%1%.png") % frameIndex).str()).string(), guiUpdate.bkgndMapImg);
		for(size_t i = 0; i < guiUpdate.oldObjModelerImgs.size(); i++) cv::imwrite((outdir / (boost::format("oldObj%1%_%2%.png") % frameIndex % i).str()).string(), guiUpdate.oldObjModelerImgs[i]);
		for(size_t i = 0; i < guiUpdate.newObjModelerImgs.size(); i++) cv::imwrite((outdir / (boost::format("newObj%1%_%2%.png") % frameIndex % i).str()).string(), guiUpdate.newObjModelerImgs[i]);
	//	cv::imwrite((outdir / (boost::format("bestposes%1%.png") % frameIndex).str()).string(), guiUpdate.targetCamPoseImg);
#endif
	}
		u.stop("visualize per frame");
#endif

		frameIndex++;
		t.stop("run one iter of outmost loop");
		t2.stop("run all iters so far");
		if(makeBkgndMapOnly && frameIndex > 20) break;
	}
	/*
	 * end main loop; the planning thread should also stop now
	 */

	/*
	 * final dumping
	 */
	modeler->saveBkgndMap(outdir / "bkgndMap");
	modeler->saveCurMap(outdir / "curMap");
	if(curMapPoseWrtRaveWorld) //save for later alignment to other maps made with the robot, assuming the base hasn't moved
	{
		ofstream outfile((outdir / "curMap" / "mapPoseWrtRaveWorld.dat").string());
		ASSERT_ALWAYS(outfile);
		outfile << xf::xform2string(curMapPoseWrtRaveWorld.get()) << endl;
	}
	const std::vector<std::shared_ptr<triangulatedMesh>> newObjMeshes = modeler->getNewModelMeshes();
	for(size_t i = 0; i < newObjMeshes.size(); i++) savePeterIntelMesh(*newObjMeshes[i], outdir / (boost::format("newObjMesh%1%.ply") % i).str());
	const std::vector<std::shared_ptr<triangulatedMesh>> oldObjMeshes = modeler->getOldModelMeshes();
	for(size_t i = 0; i < oldObjMeshes.size(); i++) savePeterIntelMesh(*oldObjMeshes[i], outdir / (boost::format("oldObjMesh%1%.ply") % i).str());
	cout << "all done" << endl;
	throw std::runtime_error("all done"); //make the program stop; dtors are called on exception throw but not on exit()
		});

	/*
	 * QtApplication::exec() must be called from the main thread
	 *
	 * if we're using openrave, the viewer requires that it instead be the main loop, and it's now qt so it'll call app.exec()
	 */
	gui->show();
	if(viewer)
	{
		std::shared_ptr<std::thread> cameraUpdateThread;
		if(robotCamHandler && false/* set to true when making a video */)
		{
			/*
			 * have viewer camera follow video camera
			 */
			cameraUpdateThread.reset(new std::thread([&]()
				{
					while(true)
					{
						const std::vector<OpenRAVE::dReal> robotConfig = robotCamHandler->getLatestConfiguration();
						const rgbd::eigen::Affine3f viewerCamPoseWrtRaveWorld = robotCamHandler->getRobotBasePoseWrtRaveWorld(robotConfig) * robotCamHandler->getCamPoseWrtRobotBase(robotConfig) * rgbd::eigen::Translation3f(.3, -.24, -2.3)/* offset from recorded cam pose */;
						viewer->SetCamera(eigenXform2raveXform(viewerCamPoseWrtRaveWorld));
						std::this_thread::sleep_for(std::chrono::milliseconds(500));
					}
				}));
		}
		else
		{
			/*
			 * initialize viewer cam to something decent
			 */
			const rgbd::eigen::Affine3f viewerCamPose = xf::camPoseFromLookAt(rgbd::eigen::Vector3f(-2.5, -2.5, .8), rgbd::eigen::Vector3f(.5, -.5, 0), rgbd::eigen::Vector3f(0, 0, 1));
			const OpenRAVE::Transform raveViewerCamPose = eigenXform2raveXform(viewerCamPose);
			viewer->SetCamera(raveViewerCamPose);
		}

		/*
		 * save rendered rave viewer frames
		 */
		const auto saveViewerRenderFunc = [&](const uint8_t* imgbuf, int width, int height, int pixeldepth)
			{
				static size_t nextFrameNum = 0;
				const fs::path outpath = outdir / (boost::format("viewerframe%1$08d.png") % nextFrameNum).str();
				ASSERT_ALWAYS(pixeldepth == 3);
				const cv::Mat cvImg(height, width, cv::DataType<cv::Vec3b>::type, const_cast<uint8_t*>(imgbuf)); //shallow copy of imgbuf
				const std::vector<int> params = {CV_IMWRITE_PNG_COMPRESSION, 9};
			//	cv::imwrite(outpath.string().c_str(), cvImg, params); //TODO this takes a LOT of disk space for long experiments -- commented 20140519 for space reasons
				nextFrameNum++;
			};
		const auto callbackHandle = viewer->RegisterViewerImageCallback(saveViewerRenderFunc); //don't delete the returned handle or it'll unregister the callback

		// finally call the viewer's infinite loop
		const bool showgui = true;
		viewer->main(showgui);
	}
	else
	{
		app->exec();
	}

	return 0;
}
