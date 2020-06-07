/*
 * onlineModeler: object modeling for an icra14 submission
 *
 * Evan Herbst
 * 7 / 13 / 13
 */

#ifndef EX_ONLINE_MODELER_H
#define EX_ONLINE_MODELER_H

#include <tuple>
#include <memory>
#include "scene_differencing/sceneDifferencingOnlineFrames.h"
#include "vrip_utils/triangulatedMesh.h"

class onlineObjModeler: public onlineSceneDifferencer
{
	public:

		onlineObjModeler(const std::shared_ptr<rgbdSensorNoiseModel>& noiseModel, const onlineSceneDifferencer::params& differParams, const bool doModeling);
		virtual ~onlineObjModeler() {}

		/*
		 * call for each frame
		 *
		 * return false on error
		 */
		virtual bool nextFrameAux(rgbdFrame& frame, const boost::posix_time::ptime& frameTime);

		/*
		 * objs new in the scene
		 *
		 * from the current camera pose
		 */
		std::vector<cv::Mat_<cv::Vec3b>> getCloseUpNewModelImgs() const;
		/*
		 * objs no longer in the scene
		 *
		 * from the current camera pose
		 */
		std::vector<cv::Mat_<cv::Vec3b>> getCloseUpOldModelImgs() const;
		/*
		 * from the orientation of the current camera but centered on the model
		 */
		std::vector<cv::Mat_<cv::Vec3b>> getNewModelImgsV2() const;
		std::vector<cv::Mat_<cv::Vec3b>> getOldModelImgsV2() const; //TODO broken 20140212?
		/*
		 * use a viewpoint that makes the object easily visible
		 */
		std::vector<cv::Mat_<cv::Vec3b>> getNewModelImgsV3() const;
		std::vector<cv::Mat_<cv::Vec3b>> getOldModelImgsV3() const;

		std::vector<std::shared_ptr<triangulatedMesh>> getNewModelMeshes() const;
		std::vector<std::shared_ptr<triangulatedMesh>> getOldModelMeshes() const;

	protected:

		/*
		 * auxiliary to createAndUpdateObjModels()
		 *
		 * mark only models whose volumes intersect the viewing frustum, so we can avoid keeping the others in gpu memory
		 */
		template <typename ObjModelT>
		std::vector<bool> listObjModelVisibility(const std::vector<ObjModelT>& objModelers, const float maxFrustumZ);

		/*
		 * auxiliary to nextFrame()
		 */
		void createAndUpdateObjModels(rgbdFrame& frame);

		/*
		 * auxiliary to nextFrame()
		 */
		void processSegmentationDemonstration(rgbdFrame& frame);

		/*
		 * for object modeling
		 */
		bool doObjModeling; //if not, this class does nothing
		struct oldObjModelerInfo
		{
			std::shared_ptr<VolumeModeler> modeler; //a copy of the map with non-object bits removed (not high-res)
			VolumeModelerAllParams params;
			uint32_t initialFrame; //which frame index of the new scene the modeler's xform is relative to
			rgbd::eigen::Affine3f initialCamPoseWrtMap;
			uint32_t lastUpdatedFrame;
		};
		struct newObjModelerInfo
		{
			std::shared_ptr<VolumeModeler> modeler; //high-res models of changed segments in the cur scene
			VolumeModelerAllParams params;
			uint32_t initialFrame; //which frame index of the new scene the modeler's xform is relative to
			rgbd::eigen::Affine3f initialCamPoseWrtMap;
			uint32_t lastUpdatedFrame;
		};
		std::vector<oldObjModelerInfo> oldObjModelers;
		std::vector<newObjModelerInfo> newObjModelers;
		std::vector<cv::Mat_<cv::Vec3b>> objModelerImgs; //visualization
		static const size_t modelInactivityDeallocationPeriod = 4; //move models that haven't been touched in this many frames off the gpu; TODO ?

		std::shared_ptr<VolumeModeler> bkgndPlusObjsModeler; //TODO experimenting 20140430

		/*
		 * for processing object demos (see icra14 submission)
		 */
		bool inDemoMode;
};

#endif //header
