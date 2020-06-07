/*
 * onlineActiveModeler: active but non-interactive object modeling for an icra14 submission
 *
 * Evan Herbst
 * 7 / 13 / 13
 */

#ifndef EX_ONLINE_ACTIVE_MODELER_H
#define EX_ONLINE_ACTIVE_MODELER_H

#include <array>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <boost/optional.hpp>
#include <opencv2/core/core.hpp>
#include <openrave-core.h>
#include <ros/node_handle.h>
#include "rgbd_util/parallelism.h"
#include "vrip_utils/voxelGrids.h"
#include "scene_differencing/generativeSensorNoiseModel.h"
#include "active_vision_common/cameraOnRobotHandler.h"
#include "active_vision_common/robotSpec.h"
#include "active_vision_common/functionOverVoxels.h"
#include "grasp_utils/ExecuteGraspTest.h" //the service to run a trajectory
#include "active_obj_modeling/collisionChecking.h"
#include "active_obj_modeling/continuousTrajSender.h"
#include "active_obj_modeling/onlineModeler.h"

class onlineActiveModeler
{
	public:

		/*
		 * we might edit differParams before we pass it along
		 *
		 * renderer must have had a large texture allocated, as we may do view scoring
		 *
		 * pre: camHandler->initROSSubscribers() has been called
		 */
		onlineActiveModeler(const std::shared_ptr<openglContext>& ctx, const std::shared_ptr<viewScoringRenderer>& renderer, const std::shared_ptr<robotSpec>& rspec, const std::shared_ptr<cameraOnRobotHandler>& camHandler,
			const std::shared_ptr<continuousTrajSender>& trajSender, const std::shared_ptr<onlineSceneDifferencer>& m, const configOptions& cfg, OpenRAVE::EnvironmentBasePtr visEnv, const rgbd::eigen::Affine3f& mapPoseWrtRaveWorld);
		virtual ~onlineActiveModeler() {}

		/*
		 * updatePerModelingFrame() needs to be called after the modeler processes each frame, while the maps mutex is still locked;
		 * then when update() runs, we'll have the results of all frames the modeler has processed so far
		 */

		/*
		 * update synchronously with modeling, after each frame of modeling
		 * (eg, so we can copy results from this frame that the mapper won't store for future frames)
		 */
		void updatePerModelingFrame(rgbdFrame& frame, const boost::posix_time::ptime& frameTime);

		/*
		 * update asynchronously from modeling, which runs in a separate thread
		 */
		void updateAsync();

		cv::Mat_<cv::Vec3b> getTargetCamPoseImg() const;
		std::string getTargetCamPoseDescription() const; //to be shown in a UI
		cv::Mat_<cv::Vec3b> getMeshValuesImg() const;

		/*
		 * return some gui-able info about the last time we abandoned a target
		 */
		std::string getTargetSwitchInfo() const;

		bool isExperimentOver() const {return experimentOver;}

	protected:

		/*
		 * auxiliary to asynchronous update
		 *
		 * extract all the data structures we need from maps for this iteration
		 */
		void processMapsBeforeViewSelection();

		/*
		 * inverse kinematics
		 *
		 * tgtPoseWrtRobotBase will be applied to all poses in the vector ('tgt' is user-defined)
		 *
		 * return an ik solution for each pose, or empty if no solution was found
		 */
		std::vector<std::vector<OpenRAVE::dReal>> runIKForPoses(const rgbd::eigen::Affine3f& tgtPoseWrtRobotBase, const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& camPosesWrtTgt,
			const bool doCollisionChecking);

		/*
		 * compute signed distance fields for all objects for orcdchomp
		 */
		void computeSDFsForCHOMP();
		/*
		 * remove stored signed distance fields from orcdchomp
		 */
		void clearSDFsFromCHOMP();

		struct viewTargetInfo
		{
			viewTargetInfo() : triIndex(-1), score(-1)
			{}

			int64_t triIndex; //into full mesh; the tri this pose looks at, or < 0 if that's not how this pose was generated
			std::array<rgbd::pt, 3> triVertices; //valid if triIndex >= 0
			float viewDist; //distance from cam pos to triangle centroid; TODO not useful?
			float score; //< 0 if not set
			rgbd::eigen::Affine3f tgtPoseWrtMap; //target surface patch: translation gives centroid; z axis of rot mtx gives surface normal
		};

		struct suggestedCamPoseInfo
		{
			viewTargetInfo targetInfo;
			boost::optional<rgbd::eigen::Affine3f> camPoseWrtMap;
			std::vector<OpenRAVE::dReal> viewingConfiguration; //empty if hasn't been calculated
		};

#ifdef UNUSED
		/*
		 * return a list of cam poses wrt the mesh
		 *
		 * pick one frontier triangle and propose many poses looking at it
		 */
		std::vector<suggestedCamPoseInfo> suggestPosesViewingNearestFrontierTriangle(const rgbd::eigen::Affine3f& curMapPoseWrtRaveWorld,
			const std::shared_ptr<triangulatedMesh>& curMesh, const std::vector<size_t>& frontierTrianglesVec);
#endif
#ifdef UNUSED
		/*
		 * return a list of cam poses wrt the mesh
		 *
		 * propose poses near the current one
		 */
		std::vector<suggestedCamPoseInfo> suggestPosesNearCurrent(const rgbd::eigen::Affine3f& curMapPoseWrtRaveWorld);
#endif

		/*
		 * return a ranked subset of all mesh triangles that we should consider looking at in order
		 *
		 * negative triangle values mean it's not useful to look at them
		 */
		std::vector<onlineActiveModeler::viewTargetInfo> suggestOrderedViewTargets(const rgbd::eigen::Affine3f& curMapPoseWrtRaveWorld,
			const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<float>& triangleValues);

		/*
		 * return a list of cam poses wrt the map
		 *
		 * negative triangle values mean it's not useful to look at them
		 */
		std::vector<suggestedCamPoseInfo> suggestPosesViewingValuableTriangles(const rgbd::eigen::Affine3f& curMapPoseWrtRaveWorld,
			const std::shared_ptr<triangulatedMesh>& curMesh, const std::vector<float>& triangleValues);

		/*
		 * render the mesh from all proposed poses and score each view
		 *
		 * score function is parameterized:
		 * - unseen triangles get a score of unseenValue
		 * - border triangles (next to unseen) get a score of borderValue
		 * - p(m) is multiplied by movednessValue and added to the score
		 *
		 * return: a score for each cam pose wrt map
		 */
		std::vector<float> scorePosesByRendering(const std::shared_ptr<triangulatedMesh>& curMesh, const std::vector<uint8_t>& triangleIsSurface, const std::unordered_set<size_t>& borderTriangles,
			const sumLogprobsCombined& meshDiffingSums, const uint8_t unseenValue, const uint8_t borderValue, const uint8_t movednessValue, const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& camPosesWrtMap);

		/*
		 * list triangles at seen/unseen borders in the mesh
		 */
		std::vector<size_t> findSurfaceTrianglesBorderingUnseen(const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<uint8_t>& triangleIsSurface) const;

		/*
		 * info for computing geodesic distances: return for each triangle the avg distance from the centroid to a vertex
		 */
		std::vector<float> getMeshTriangleSizes(const triangulatedMesh& mesh) const;

		/*
		 * info for computing geodesic distances: return for each triangle the indices of all neighboring triangles, unsorted
		 */
		std::vector<std::vector<size_t>> getTriangleNeighbors(const triangulatedMesh& mesh) const;

		/*
		 * return an approximate distance, in meters, of each triangle along mesh surfaces to the nearest high-p(m) triangle
		 *
		 * maxDist: only find triangles up to this far (in m) from moved surfaces; return some very large distance for other triangles
		 */
		std::vector<float> getUnseenTrianglesGeodesicDistToMovedSurface(const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<float>& triSizes, const std::vector<std::vector<size_t>>& triNbrs,
			const sumLogprobsCombined& meshDiffingSums, const float maxDist) const;

		/*
		 * return: approximate traversal cost, in meter-equivalents, from cur cam pose to each provided pose
		 */
		std::vector<float> getTraversalCostsToCamPoses(const std::shared_ptr<triangulatedMesh>& curMapMesh, const std::vector<OpenRAVE::dReal>& curConfig,
			const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& camPosesWrtMap);

		/*
		 * get differencing info per mesh triangle in the current map
		 */
		sumLogprobsCombined getAggregatedDiffingResultsPerCurMeshTriangle(const std::shared_ptr<triangulatedMesh>& curMapMesh);

		std::shared_ptr<precomputedEnvCollisionData> precomputeEnvironmentGeometryForCollisionChecking();

		struct camPoseSuggestionData;
		/*
		 * return: suggested next camera poses, most useful first (might be an empty list), with some sort of scores;
		 * return only a few at a time so they don't all have to be tested for reachability if one of the first few will be reachable, which is usually the case
		 *
		 * if geometricEnvData isn't empty, we'll copy it into the result structure
		 *
		 * post: each pose suggestion has triIndex set but doesn't have a pose yet
		 */
		std::shared_ptr<camPoseSuggestionData> initCamPoseSuggestion(const std::shared_ptr<precomputedEnvCollisionData>& geometricEnvData);
		/*
		 * return more camera poses/robot configurations to try planning to, or empty if we're out of ideas
		 *
		 * return no more than maxSuggestionsAtOnce suggestions at a time (this function is meant to be called until an acceptable suggestion is found)
		 */
		std::vector<onlineActiveModeler::suggestedCamPoseInfo> suggestMoreCamPoses(const std::shared_ptr<camPoseSuggestionData>& data, const size_t maxSuggestionsAtOnce = 1000000);

		/*
		 * for keeping track of map areas we want to avoid looking at in future
		 */
		enum class viewsByPoseOri : uint8_t {POSX, NEGX, POSY, NEGY, POSZ, NEGZ}; //where the camera is wrt the look-at pt
		struct poseQuantizationIndex
		{
			std::array<int64_t, 3> voxel;
			viewsByPoseOri ori;
		};
		poseQuantizationIndex getPoseQuantizationIndex(const std::shared_ptr<triangulatedMesh>& curMapMesh, const suggestedCamPoseInfo& suggestedPose, const VolumeModelerAllParams& mapperParams) const;

		/*
		 * resample the traj so the temporal and spatial distance between each pair of waypts is what we think mapping can handle in one iteration (in order to not break the small-movement assumption)
		 *
		 * deltaT: time taken between each consecutive pair of samples
		 */
		void retimeTrajectoryForMapping(std::vector<std::vector<OpenRAVE::dReal>>& trajSamples, OpenRAVE::dReal& deltaT);

		/*
		 * represent the output of planning
		 */
		struct motionPlanInfo
		{
			std::vector<std::vector<OpenRAVE::dReal>> waypoints; //empty iff planning failed
			OpenRAVE::dReal deltaT; //between each pair of consecutive waypoints
			bool reachesTarget; //does this plan get us all the way to the current target cam pose?
		};

		/*
		 * return whether we're currently (i.e. on the current frame) using the CHOMP motion planner
		 */
		bool usingChomp() const;

		/*
		 * plan a motion using simple configuration interpolation
		 *
		 * ideally usable with any type of robot
		 *
		 * intended to be used only for small motions
		 */
		motionPlanInfo planPathByConfigInterpolation(const std::vector<OpenRAVE::dReal>& planStartConfig, const std::vector<OpenRAVE::dReal>& targetConfig);

		/*
		 * plan a motion for a kinematic-chain robot using a planner in openrave
		 */
		motionPlanInfo planKinematicChainPathWithOpenRAVE(const std::vector<OpenRAVE::dReal>& planStartConfig, const std::vector<OpenRAVE::dReal>& targetConfiguration, const OpenRAVE::EnvironmentBasePtr& planningEnv);

		/*
		 * plan a motion for a free-flying robot using a tree-based planner (eg an rrt)
		 */
		motionPlanInfo planFreeFlyingPathWithTreeBasedPlanner(const std::vector<OpenRAVE::dReal>& planStartConfig, const std::vector<OpenRAVE::dReal>& targetConfig);

		/*
		 * run motion planning
		 *
		 * targetCamPoseInfo.viewingConfiguration, if not empty, should be a valid solution for the given camera pose
		 *
		 * if planning fails, return an empty list of waypoints
		 */
		motionPlanInfo planPath(const suggestedCamPoseInfo& targetCamPoseInfo, const std::shared_ptr<precomputedEnvCollisionData>& collisionCheckingEnvData);

		/*
		 * visualize the traj and send it to the robot
		 */
		void enqueueTrajectory(const motionPlanInfo& plan);

		/*
		 * scores in [0, 1], 1 best
		 *
		 * if an outpath is empty, don't write the corresponding file
		 */
		cv::Mat_<cv::Vec3b> visualizeMeshWithCameraPoses(const triangulatedMesh& mesh, const std::vector<std::pair<rgbd::eigen::Affine3f, float>>& camPosesWrtMapWithScores,
			const rgbd::eigen::Affine3f& viewingPoseWrtInitialCamPose, const bool showCurCamPose, const fs::path& imgOutpath, const fs::path& meshOutpath = fs::path()) const;

		/*
		 * visualize robot + environment as meshes
		 */
		void visualizeRobotPoses(const std::shared_ptr<triangulatedMesh>& envMesh, const std::vector<std::vector<OpenRAVE::dReal>>& configurations, robotSpec& robotInterface, const fs::path& outdirPlusFilebase) const;

		/*
		 * read tsdf values
		 *
		 * return the value for unseen if pos is outside map bounds
		 */
		float tsdfValueAtPos(const rgbd::eigen::Vector3f& posWrtRaveWorld) const;
		/*
		 * read tsdf values
		 *
		 * return whether the map says the given position is known free (this includes a check for it being outside map bounds)
		 */
		bool tsdfFreeAtPos(const rgbd::eigen::Vector3f& posWrtRaveWorld) const;

		/******************************
		 * an interface layer to abstract away the type of robot (flying, kinematic-chain, wheeled...)
		 */

		/*
		 * for each input pose, propose a configuration of the robot that allow its camera to reach that pose, or return an empty configuration if we can't find one
		 */
		std::vector<std::vector<OpenRAVE::dReal>> getViewingConfigurationsForPoses(const rgbd::eigen::Affine3f& tgtPoseWrtRaveWorld, const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& camPosesWrtTgt,
			const bool doCollisionChecking);

		/*
		 * return: pose distance from the given pose wrt the cur cam pose to, for each mesh triangle, a cam pose viewing that triangle
		 *
		 * daWeight: weight for angular dist vs. position dist used in pose dist; I usually use around .5
		 */
		std::vector<float> getPoseDistToCamPosesViewingMeshTriangles(const rgbd::eigen::Affine3f& poseWrtCamToConsider, const float daWeight = .4);

		std::vector<uint8_t> checkRobotCollisions(const std::shared_ptr<precomputedEnvCollisionData>& collisionCheckingEnvData, const std::vector<std::vector<OpenRAVE::dReal>>& configurations);

		/*****************************/

		/*
		 * do we use said info for anything? if not, don't update the per-voxel diffing-results map each frame
		 */
		bool shouldKeepDiffingInfoForCurScene() const;

		std::shared_ptr<robotSpec> robotInterface;
		rgbd::cameraSetup cams;

		std::shared_ptr<onlineSceneDifferencer> onlineMapper;

		std::shared_ptr<openglContext> glContext;
		std::shared_ptr<viewScoringRenderer> sceneRenderer;

		/*
		 * duplicated from onlineSceneDifferencer so we can avoid reading while it's updating
		 * (we update these each time we run)
		 */
		uint64_t frameIndex;
		std::vector<rgbd::eigen::Affine3f> framePosesWrtCurMap;

		/*
		 * for controlling camera motion
		 */
		bool inSimulation; //whether we're doing everything in openrave
		bool moveRobot; //if not, we won't try to move the camera
		std::shared_ptr<cameraOnRobotHandler> robotCamHandler; //not set if not running on robot

		OpenRAVE::EnvironmentBasePtr visEnv; //for visualization; also we generally assume it contains most interesting kinbodies

		rgbd::eigen::Affine3f curMapPoseWrtRaveWorld; //"cur" means current as opposed to prev scene--this is constant during the new scene
		boost::optional<rgbd::eigen::Affine3f> targetCamPose; //if empty, there's currently no goal
		mutable cv::Mat_<cv::Vec3b> targetCamPoseImg; //visualization
		std::shared_ptr<rgbd::thread> openraveViewerThread; //to run a viewer's main loop

		/*
		 * for assigning a viewing value to each unseen mesh triangle
		 */
		enum class triValueFuncType : uint8_t
		{
			RANDOM_FRONTIER,
			DISTANCE_TO_CAM, //distance to cur cam pose
			DISTANCE_TO_CUR_TARGET, //distance to what's currently seen in the middle of the image (NOT to the view target)
			DISTANCE_TO_MOVED_SURFACE, //distance to high-p(m) surface
			DISTANCE_HYBRID_TARGET_MOVED, //combination of distance to cur target and distance to high-p(m) surface
			DISTANCE_HYBRID_CAM_SURFACE, //combination of distance to cam and distance to any surface (dieter likes this one -- it's an attempt to upweight looking at objs on a table first) -- TODO possibly superseded by HYBRID_TARGET_SURFACE
			DISTANCE_HYBRID_TARGET_SURFACE, //combination of distance to cur target and distance to any surface (dieter likes this one -- it's an attempt to upweight looking at objs on a table first)
			INFO_GAIN, //an approximation to information gain, using info from all triangles seen
			TRAVERSAL_COST_FRONTIER, //select a frontier triangle using only traversal cost
			DISTANCE_HYBRID_INFOGAIN_TRAVERSALCOST
		};
		triValueFuncType triValueFunc;

		triValueFuncType str2triValueFunc(const std::string& s) const;

		/*
		 * keep track of areas we've tried to look at and failed to map (useful, eg, with surfaces that won't return kinect readings and with surfaces that, due to mapping drift, become hidden in the map)
		 *
		 * voxel index -> which voxel plane we're looking at {+x, -x, +y, -y, +z, -z} (a rough quantization of orientation) -> how many times we've already viewed this pose
		 *
		 * cover the same volume as the scene map
		 */
		functionOverVoxels<std::array<uint16_t, 6>> viewsByPose;
		static constexpr size_t maxViewsPerQuantizedPose = 1; //after this many plus one, we don't try to look at this map bit again, but instead assume it's not measurable; TODO ?
		std::vector<std::array<int64_t, 4>> overviewedViews; //list of ones we've seen a lot
		std::vector<std::array<int64_t, 4>> allViewsPlannedTo; //list of ones we've seen at all; only needed for visualization

		/*
		 * estimates at how much or how long something will take at runtime
		 */
		const float mappingMaxIterationDX = .02, //max pose distance we think one mapping iteration can handle if change is only in translation (Peter guesses 5 cm is reasonable)
			mappingMaxIterationDA = .07; //max pose distance we think one mapping iteration can handle if change is only in rotation (Peter has no idea what's reasonable)
		//educated guess at how long we'll take to run one mapping outer loop iter; TODO continually refresh during runtime w/ max taken so far, or something conservative like that?
		float estimatedMappingIterationRuntime;
		float estimatedTrajPlanTime = 2; //educated guess at how long it will take to plan the next trajectory, in s; TODO ?; also depends on what type of planning we're doing, eg gradient vs full path

		OpenRAVE::ModuleBasePtr chompModule; //CHOMP motion planning
		std::vector<OpenRAVE::dReal> planStartConfig; //where to start the next path plan from (should usually be the end of the previously made plan)
		bool resetPlanStartConfig; //whether to start planning from the current robot configuration next time rather than the end of the previously made plan, due to some motion execution failure
		std::mutex planStartConfigMux; //guard the reset flag only

		/*
		 * a target is a camera pose chosen by view suggestion; it takes one or more plans to reach a target
		 */
		std::shared_ptr<suggestedCamPoseInfo> curTarget; //when empty, there is no cur view target
		int64_t nextTargetIDToUse; //an id given to the trajectory executor; increasing over time starting at 0
		int64_t curTargetID; //valid when curTarget isn't empty
		int64_t lastTargetReached;
		int64_t nextPlanIDToUse;
		std::unordered_map<int64_t, int64_t> plansReachingTargets; //id of a plan that reaches a target -> target id (plans not reaching targets aren't in here)
		std::unordered_set<int64_t> plansCompleted; //written by the traj sender, read by us
		std::mutex plansCompletedMux;
		std::shared_ptr<suggestedCamPoseInfo> lastTarget; //last target chosen, if any has been chosen (i.e., this is almost always defined)
		//for debugging/visualization
		std::string targetSwitchStateStr; //info about what happened when we last switched targets

		/*
		 * for motion execution
		 */
		std::shared_ptr<continuousTrajSender> trajSender; //send motion commands to a real or simulated robot
		std::shared_ptr<std::thread> trajSenderThread;
		timespec modelingStartTime; //set in ctor; for comparison to end times of previously planned trajs

		/*
		 * the map
		 */

		std::shared_ptr<triangulatedMesh> curMapMesh; //map mesh including unseen triangles, in the coord frame of the map
		std::vector<uint8_t> triangleIsSurface; //per triangle in curMapMesh; 0 = unseen, 1 = seen
		std::shared_ptr<triangulatedMesh> curMapMeshSeenOnly; //map mesh including only seen triangles (ie surfaces), in the coord frame of the map

		//for holding peter-intel tsdf buffers
		std::vector<boost::shared_ptr<std::vector<float>>> bufferDVectors;
		std::vector<boost::shared_ptr<std::vector<float>>> bufferDWeightVectors;
		std::vector<boost::shared_ptr<std::vector<unsigned char>>> bufferCVectors;
		std::vector<boost::shared_ptr<std::vector<float>>> bufferCWeightVectors;
		std::vector<boost::shared_ptr<Eigen::Array3i>> bufferVoxelCounts;
		std::vector<boost::shared_ptr<Eigen::Affine3f>> tsdfBufferPosesWrtMap;

		/*
		 * differencing results on the cur map
		 */
		struct voxelDiffingInfo
		{
			voxelDiffingInfo() : evidenceMoved(0), evidenceNotMoved(0)
			{}

			float evidenceMoved, evidenceNotMoved;
		};
		functionOverVoxels<voxelDiffingInfo> diffingResultWrtCurMap; //in the same coord frame as that of the cur map

		/*
		 * for assigning values to mesh triangles
		 */
		cv::Mat_<cv::Vec3b> meshValuesImg; //visualization

		/*
		 * for collision checking
		 */
		//simplified meshes of the robot, to speed up collision checking wrt using all triangles
		std::vector<triangulatedMesh> simplifiedRobotLinkMeshes;
		std::vector<uint32_t> robotLinkMeshLinkIndices; //which link each mesh comes from
		std::vector<std::shared_ptr<triangulatedMesh>> staticRaveEnvMeshes; //things the robot could collide with apart from the changing output of mapping
		//std::shared_ptr<precomputedEnvCollisionData> collisionCheckingEnvData; //computed once at the beginning of each view selection process

		//extra openrave robots to speed up reachability checking and motion planning
		std::vector<OpenRAVE::EnvironmentBasePtr> raveEnvCopies;
		OpenRAVE::EnvironmentBasePtr planningEnv;

		fs::path outdir;

		/*
		 * experiment statistics
		 */
		bool experimentOver; //for different experiments
		float distanceTraveled; //updated whenever we call the trajSender
		size_t numPlans, numSuccessfulPlans;
		size_t numPoseSuggestionsComputed, numPoseSuggestionsDiscarded;
		int32_t lastTargetReachedByExistingPlan;
		float totalUpdateTime; //for runs of updateAsync
		size_t updateCount;
		voxelGrid<uint8_t> initialSceneFreeVoxels; //whether each voxel in the map is free in the true scene (1 = free, 0 = nonfree, other values are errors) (to be compared to which voxels are free in the built map at different points in time)
		bool trackKnownFreeSpace; //whether to track the % of initialSceneFreeVoxels that are known free during active mapping
};

#endif //header
