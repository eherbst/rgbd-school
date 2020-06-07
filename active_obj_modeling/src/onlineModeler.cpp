/*
 * onlineModeler: object modeling for an icra14 submission
 *
 * Evan Herbst
 * 7 / 13 / 13
 */

#include "pcl_rgbd/cloudTofroPLY.h"

#include <unordered_set>
#include <iostream>
#include <fstream>
#include <boost/pending/disjoint_sets.hpp>
#include <CGAL/version.h>
#if CGAL_VERSION_NR < 1040100000
#error need cgal >= 4.1 for polyhedron-polyhedron intersection testing (intersection_of_Polyhedra_3.h)
//(I've tried the Nef polyhedron stuff that's supported in all recent versions; there's weirdness with the kernels it requires)
#endif
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/intersection_of_Polyhedra_3.h>
#include <CGAL/IO/Polyhedron_VRML_2_ostream.h>
#include "xforms/xforms.h"
#include "rgbd_depthmaps/depthIO.h"
#include "rgbd_ros_util/ros_utility.h"
#include "cuda_util/cudaUtils.h"
#include "scene_rendering/triangulatedMeshRenderer.h"
#include "scene_rendering/sceneSamplerRenderedTSDF.h"
#include "scene_rendering/castRaysIntoSurfels.h"
#include "rgbd_frame_common/rgbdFrame.h"
#include "rgbd_frame_common/staticDepthNoiseModeling.h"
#include "peter_intel_mapping_utils/rendering.h"
#include "peter_intel_mapping_utils/conversions.h"
#include "sm_common/renderSmallCloudNicely.h"
#include "scene_differencing/approxRGBDSensorNoiseModel.h"
#include "scene_differencing/visualization.h"
#include "active_obj_modeling/diffingConnComps.h"
#include "active_obj_modeling/onlineModeler.h"
using std::vector;
using std::cout;
using std::endl;
using std::ofstream;
using rgbd::eigen::Vector3f;

#define OLD_MODELS_FROM_FRAMES //use cur frames to add to old-obj models rather than taking them from the prev map (ideally we want lots of info from the prev map but it's harder to make models look nice)

typedef CGAL::Exact_predicates_inexact_constructions_kernel  Kernel;
typedef CGAL::Polyhedron_3<Kernel>                     Polyhedron_3;
typedef Kernel::Point_3                                Point_3;

onlineObjModeler::onlineObjModeler(const std::shared_ptr<rgbdSensorNoiseModel>& noiseModel, const onlineSceneDifferencer::params& differParams, const bool doModeling)
: onlineSceneDifferencer(noiseModel, differParams)
{
	doObjModeling = doModeling;
	inDemoMode = false;

	bkgndPlusObjsModeler.reset(new VolumeModeler(clKernels, prevSceneMapParams));

	/*
	 * try to make icra14A1n2 not die around frame 332 when there aren't enough valid bkgnd-map pixels for alignment to be happy:
	 * add cur-scene objs to the copy of the bkgnd map that's used to align new frames against (only add them for alignment: the here-newly-created modeler gets used only for alignment for one frame)
	 */
	setCurModelToAlignToGetter([this]()
		{
			cout << "before merging many models: vram total " << getNVIDIAVRAMTotal() << ", free " << getNVIDIAVRAMFree() << endl;
			sceneInfo& scene1 = *scene1ptr;
			sceneInfo& scene2 = *scene2ptr;
			//bkgndPlusObjsModeler.reset(bkgndMapper->clone());  //TODO using clone() here causes a vram leak, but the problem isn't just clone() -- see testModelerMemLeak.cpp: clone()ing in a loop doesn't waste vram! what's up?
			//bkgndPlusObjsModeler.reset(new VolumeModeler(clKernels, prevSceneMapParams));
			bkgndPlusObjsModeler->setValueInBox(-1, 0, rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(-5, -5, -5)) * rgbd::eigen::Affine3f(10 * rgbd::eigen::Matrix3f::Identity())); //set everything in the map to unseen (TODO do this non-hackily so won't break if the map is bigger than the size we use here)
			bkgndPlusObjsModeler->mergeOtherVolume(*bkgndMapper, rgbd::eigen::Affine3f::Identity());
#if 1
			for(newObjModelerInfo& modelerInfo : newObjModelers)
			{
				const rgbd::eigen::Affine3f tmpModelPoseWrtObjModel = mapPosesWrtGlobal[scene1.sceneName].inverse() * mapPosesWrtGlobal[scene2.sceneName] * newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]] * modelerInfo.initialCamPoseWrtMap.inverse();
				bkgndPlusObjsModeler->mergeOtherVolume(*modelerInfo.modeler, tmpModelPoseWrtObjModel);
#if 0
				/*
				 * the merging operation might bring a model into vram, so move it out again just in case (this would be much better done with a memory manager called from within model data structure code)
				 */
				modelerInfo.modeler->deallocateBuffers();
#endif
			}
#endif
			cout << "after merging many models: vram total " << getNVIDIAVRAMTotal() << ", free " << getNVIDIAVRAMFree() << endl;
			return bkgndPlusObjsModeler;
		});
}

/*
 * state after the last frame processed
 */

/*
 * objs new in the scene
 *
 * from the current camera pose
 */
std::vector<cv::Mat_<cv::Vec3b>> onlineObjModeler::getCloseUpNewModelImgs() const
{
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	std::vector<cv::Mat_<cv::Vec3b>> imgs(newObjModelers.size());
	for(size_t i = 0; i < newObjModelers.size(); i++)
	{
		if(newObjModelers[i].lastUpdatedFrame + modelInactivityDeallocationPeriod < frameIndex)
			imgs[i] = cv::Mat_<cv::Vec3b>(camParams.yRes, camParams.xRes, cv::Vec3b(255, 229, 204)); //fill w/ bkgnd col
		else
			imgs[i] = getPrettyIntelMapRender(*newObjModelers[i].modeler, camParams, newObjModelers[i].initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[newObjModelers[i].initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes.back()], std::array<uint8_t, 3>{{204, 229, 255}});
	}
	return imgs;
}
/*
 * from the orientation of the current camera but centered on the model
 */
std::vector<cv::Mat_<cv::Vec3b>> onlineObjModeler::getNewModelImgsV2() const
{
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	std::vector<cv::Mat_<cv::Vec3b>> imgs(newObjModelers.size());
	for(size_t i = 0; i < newObjModelers.size(); i++)
	{
		const rgbd::eigen::Vector3f modelCenterInModelCoords = rgbd::eigen::Vector3f(.5 * newObjModelers[i].params.volume.cell_size * newObjModelers[i].params.volume.cell_count.cast<float>());
#if 0 //works
		const rgbd::eigen::Affine3f fullPose = newObjModelers[i].initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[newObjModelers[i].initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes.back()];
		const rgbd::eigen::Vector3f lookAtPt = modelCenterInModelCoords;
		const rgbd::eigen::Vector3f camPos = lookAtPt + rgbd::eigen::Vector3f(0, 0, -1);//fullPose * rgbd::eigen::Vector3f::Zero();
		const rgbd::eigen::Vector3f viewDir = (lookAtPt - camPos).normalized();
		rgbd::eigen::Vector3f upDir = fullPose.linear() * rgbd::eigen::Vector3f(0, -1, 0);
		upDir = (upDir - upDir.dot(viewDir) * viewDir).normalized();
		const rgbd::eigen::Affine3f camPoseWrtModel = xf::camPoseFromLookAt(lookAtPt - .6 * viewDir.normalized(), lookAtPt, upDir);
#else
		const rgbd::eigen::Affine3f rotation((newObjModelers[i].initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[newObjModelers[i].initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes.back()]).linear());
		const rgbd::eigen::Affine3f camPoseWrtModel = rgbd::eigen::Translation3f(modelCenterInModelCoords) * rotation * rgbd::eigen::Translation3f(-.6 * rgbd::eigen::Vector3f::UnitZ());
#endif
		imgs[i] = getPrettyIntelMapRender(*newObjModelers[i].modeler, camParams, camPoseWrtModel, std::array<uint8_t, 3>{{204, 229, 255}});
	}
	return imgs;
}
/*
 * use a viewpoint that makes the object easily visible
 */
std::vector<cv::Mat_<cv::Vec3b>> onlineObjModeler::getNewModelImgsV3() const
{
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	std::vector<cv::Mat_<cv::Vec3b>> imgs(newObjModelers.size());
	glContext->acquire();
	for(size_t i = 0; i < newObjModelers.size(); i++)
	{
		const std::shared_ptr<triangulatedMesh> meshptr = generateIntelMapMesh(*newObjModelers[i].modeler);
		imgs[i] = renderSmallMeshNicely(camParams, *meshptr, *sceneRenderer);
	}
	glContext->release();
	return imgs;
}

/*
 * objs no longer in the scene
 *
 * from the current camera pose
 */
std::vector<cv::Mat_<cv::Vec3b>> onlineObjModeler::getCloseUpOldModelImgs() const
{
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	std::vector<cv::Mat_<cv::Vec3b>> imgs(oldObjModelers.size());
	for(size_t i = 0; i < oldObjModelers.size(); i++)
	{
		if(oldObjModelers[i].lastUpdatedFrame + modelInactivityDeallocationPeriod < frameIndex)
			imgs[i] = cv::Mat_<cv::Vec3b>(camParams.yRes, camParams.xRes, cv::Vec3b(255, 229, 204)); //fill w/ bkgnd col
		else
			imgs[i] = getPrettyIntelMapRender(*oldObjModelers[i].modeler, camParams, oldObjModelers[i].initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[oldObjModelers[i].initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes.back()], std::array<uint8_t, 3>{{204, 229, 255}});
	}
	return imgs;
}
/*
 * from the orientation of the current camera but centered on the model
 */
std::vector<cv::Mat_<cv::Vec3b>> onlineObjModeler::getOldModelImgsV2() const
{
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	std::vector<cv::Mat_<cv::Vec3b>> imgs(oldObjModelers.size());
	for(size_t i = 0; i < oldObjModelers.size(); i++)
	{
		const rgbd::eigen::Vector3f modelCenterInModelCoords = rgbd::eigen::Vector3f(.5 * oldObjModelers[i].params.volume.cell_size * oldObjModelers[i].params.volume.cell_count.cast<float>());
		const rgbd::eigen::Affine3f rotation((oldObjModelers[i].initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[oldObjModelers[i].initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes.back()]).linear());
		const rgbd::eigen::Affine3f camPoseWrtModel = rgbd::eigen::Translation3f(modelCenterInModelCoords) * rotation * rgbd::eigen::Translation3f(-.6 * rgbd::eigen::Vector3f::UnitZ());
		imgs[i] = getPrettyIntelMapRender(*oldObjModelers[i].modeler, camParams, camPoseWrtModel, std::array<uint8_t, 3>{{204, 229, 255}});
	}
	return imgs;
}
/*
 * use a viewpoint that makes the object easily visible
 */
std::vector<cv::Mat_<cv::Vec3b>> onlineObjModeler::getOldModelImgsV3() const
{
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	std::vector<cv::Mat_<cv::Vec3b>> imgs(oldObjModelers.size());
	glContext->acquire();
	for(size_t i = 0; i < oldObjModelers.size(); i++)
	{
		const std::shared_ptr<triangulatedMesh> meshptr = generateIntelMapMesh(*oldObjModelers[i].modeler);
		imgs[i] = renderSmallMeshNicely(camParams, *meshptr, *sceneRenderer);
	}
	glContext->release();
	return imgs;
}

std::vector<std::shared_ptr<triangulatedMesh>> onlineObjModeler::getNewModelMeshes() const
{
	std::vector<std::shared_ptr<triangulatedMesh>> meshes(newObjModelers.size());
	for(size_t i = 0; i < newObjModelers.size(); i++) meshes[i] = generateIntelMapMesh(*newObjModelers[i].modeler);
	return meshes;
}
std::vector<std::shared_ptr<triangulatedMesh>> onlineObjModeler::getOldModelMeshes() const
{
	std::vector<std::shared_ptr<triangulatedMesh>> meshes(oldObjModelers.size());
	for(size_t i = 0; i < oldObjModelers.size(); i++) meshes[i] = generateIntelMapMesh(*oldObjModelers[i].modeler);
	return meshes;
}

/*
 * auxiliary to createAndUpdateObjModels()
 *
 * mark only models whose volumes intersect the viewing frustum, so we can avoid keeping the others in gpu memory
 */
template <typename ObjModelT>
std::vector<bool> onlineObjModeler::listObjModelVisibility(const std::vector<ObjModelT>& objModelers, const float maxFrustumZ)
{
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	std::vector<bool> modelIsVisible(objModelers.size(), false);

	const float frustumXMin = (0 - camParams.centerX) / camParams.focalLength * maxFrustumZ, frustumXMax = (camParams.xRes - 1 - camParams.centerX) / camParams.focalLength * maxFrustumZ,
		frustumYMin = (0 - camParams.centerY) / camParams.focalLength * maxFrustumZ, frustumYMax = (camParams.yRes - 1 - camParams.centerY) / camParams.focalLength * maxFrustumZ;
	const std::vector<Point_3> frustumCGALPts = {Point_3(0, 0, 0), Point_3(frustumXMin, frustumYMin, maxFrustumZ), Point_3(frustumXMax, frustumYMin, maxFrustumZ), Point_3(frustumXMin, frustumYMax, maxFrustumZ), Point_3(frustumXMax, frustumYMax, maxFrustumZ)}; //in the camera's frame
	Polyhedron_3 cgalFrustum;
	CGAL::convex_hull_3(frustumCGALPts.begin(), frustumCGALPts.end(), cgalFrustum);
	for(size_t m = 0; m < objModelers.size(); m++)
	{
		cout << "testing model " << m << endl;
		const ObjModelT& modelerInfo = objModelers[m];

		/*
		 * project the model's corners into the frame and test whether any of them is in the viewing frustum
		 */
		ASSERT_ALWAYS(frameTimes.size() > frameIndex);
		ASSERT_ALWAYS(frameTimes.size() > modelerInfo.initialFrame);
		ASSERT_ALWAYS(newFramePosesWrtCurMap.find(frameTimes[frameIndex]) != newFramePosesWrtCurMap.end());
		ASSERT_ALWAYS(newFramePosesWrtCurMap.find(frameTimes[modelerInfo.initialFrame]) != newFramePosesWrtCurMap.end());
		cout << "frame xform for t=" << rgbd::convert_timestamp_to_string(frameTimes[frameIndex]) << ": " << endl << newFramePosesWrtCurMap[frameTimes[frameIndex]].matrix() << endl;
		cout << " model xforms:" << endl << (newFramePosesWrtCurMap[frameTimes[frameIndex]].inverse()).matrix() << endl << endl << (newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]]).matrix() << endl << endl << modelerInfo.initialCamPoseWrtMap.matrix() << endl << endl;
		const rgbd::eigen::Affine3f modelPoseWrtCurFrame = newFramePosesWrtCurMap[frameTimes[frameIndex]].inverse() * newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]] * modelerInfo.initialCamPoseWrtMap.inverse();
		std::vector<Point_3> modelCGALCorners(8);
		for(size_t i = 0; i < 8; i++) //encode which corner of the volume we're at
		{
			//get a corner of the model volume
			rgbd::eigen::Vector3f modelMapCoords;
			for(size_t m = 0; m < 3; m++) modelMapCoords[m] = ((i >> m) & 1) * modelerInfo.params.volume.cell_count[m] * modelerInfo.params.volume.cell_size;

			const rgbd::eigen::Vector3f curFrameCoords = modelPoseWrtCurFrame * modelMapCoords; //in the current camera's frame
			modelCGALCorners[i] = Point_3(curFrameCoords.x(), curFrameCoords.y(), curFrameCoords.z());
			//transform into projective coordinates, in which the frustum is just an AABB
			const float u = camParams.centerX + curFrameCoords[0] / curFrameCoords[2] * camParams.focalLength,
				v = camParams.centerY + curFrameCoords[1] / curFrameCoords[2] * camParams.focalLength,
				z = curFrameCoords[2];
			cout << " " << modelMapCoords.transpose() << " -> " << curFrameCoords.transpose() << " -> " << u << ' ' << v << ' ' << z << endl;
			if(u >= 0 && u <= camParams.xRes && v >= 0 && v <= camParams.yRes && z >= 0 && z <= maxFrustumZ)
			{
				modelIsVisible[m] = true;
				break;
			}
		}

		/*
		 * if no model bbox corner is in the frustum, it's still possible for the bbox to overlap the frustum; test this by intersecting the polyhedra's surfaces (not their volumes--this is much quicker and I couldn't get that to build; TODO?)
		 */
		if(!modelIsVisible[m])
		{
			Polyhedron_3 cgalModelBbox;
			CGAL::convex_hull_3(modelCGALCorners.begin(), modelCGALCorners.end(), cgalModelBbox);
			std::vector<std::vector<Point_3>> polylines;
			intersection_Polyhedron_3_Polyhedron_3(cgalFrustum, cgalModelBbox, std::back_inserter(polylines));
			if(!polylines.empty()) modelIsVisible[m] = true;
		}

		/*
		 * TODO handle the case where the frustum is completely enclosed by the model bbox
		 */
	}

	return modelIsVisible;
}

/*
 * auxiliary to nextFrame()
 */
void onlineObjModeler::createAndUpdateObjModels(rgbdFrame& frame)
{
	rgbd::timer t;
	const rgbd::CameraParams camParams = primesensor::getColorCamParams(cams.cam);
	const cv::Mat_<float>& depth = frame.getDepthImg();

	/*
	 * magic constants
	 */
	const float probMovedThreshold = .52, probNonmovedThreshold = .48; //TODO ?
	const size_t minObjCompSizeInPixels = 1000;//2000/* pixels; TODO ? */;
	const float maxFrustumZ = 10; //meters, for intersecting the camera frustum with model bboxes; TODO ?
	const float edgePixelPctThreshold = .13; //max % of pixels in a 2-d segment that can be seg-edge pixels for it to be added to an obj; TODO ?
	const float maxOldObjSameCompDist = .017; //meters; TODO ?; TODO should depend on bkgnd-map resolution (does it really need to be as high as .017 for a 1-cm-resolution bkgnd map on objsmove2?)

	/*
	 * more data structures of the rendered bkgnd map
	 */
	cv::Mat_<float> renderedBkgndDepth(camParams.yRes, camParams.xRes);
	cv::Mat_<cv::Vec3b> renderedBkgndCols(camParams.yRes, camParams.xRes);
	for(size_t i = 0, l = 0; i < camParams.yRes; i++)
		for(size_t j = 0; j < camParams.xRes; j++, l++)
			if((*prevRenderingValidity)[l])
			{
				renderedBkgndDepth(i, j) = (*prevRenderedPts)[l * 4 + 2];
				const cv::Vec4b bgra = prevRenderedCols.at<cv::Vec4b>(i, j);
				renderedBkgndCols(i, j) = cv::Vec3b(bgra[0], bgra[1], bgra[2]);
			}
			else
			{
				renderedBkgndDepth(i, j) = 0;
				renderedBkgndCols(i, j) = cv::Vec3b(0, 0, 0); //just for visualization
			}

	/*
	 * for diffing obj models vs cur frame
	 */

	const float posWeight = 1, colWeight = 0/*.5*/, normalWeight = 0; //TODO ?

	diffingSingleFrameInfo scene2info;
	scene2info.sceneName = "scene2";
	scene2info.frameDepth = frame.getDepthImg();
	scene2info.depthStdevs.resize(boost::extents[camParams.yRes][camParams.xRes]);
	scene2info.depthStdevs = frame.getDepthUncertainty();
	scene2info.frameImg = frame.getColorImg();
	scene2info.frameNormals = frame.getNormals();
	scene2info.framePoseWrtMap = rgbd::eigen::Affine3f::Identity();

	/*************************************************************************************************************************************************************************************************************************************************
	 * create and update models of objects that appear in the old scene and not in the new
	 */
{
	t.restart();
	sceneInfo& scene1 = *scene1ptr;
	sceneInfo& scene2 = *scene2ptr;

	/*
	 * find conn comps of high-p(m) points in the bkgnd scene wrt the cur scene
	 */
	const rgbd::eigen::VectorXd movedProbsOld = lastFrameLogprobSumsWrtFrame.movedProbs();
	std::unordered_map<size_t, std::vector<size_t>> compsByRepOld = findDiffingConnComps(camParams, renderedBkgndDepth, movedProbsOld, probMovedThreshold, diffingConnCompsAlgorithm::LARGE_NBRHOOD_2D, .01);
	//TODO split comps using previously obtained segm info from existing models if those models are certain about their outlines

	const size_t oldObjModelersSizeBeforeCreating = oldObjModelers.size(); //so we won't look for diffing results wrt modelers we've created during the same frame

	/*
	 * list only models whose volumes intersect the viewing frustum, so we can avoid keeping the others in gpu memory
	 */
	const std::vector<bool> modelIsVisible = listObjModelVisibility(oldObjModelers, maxFrustumZ);
	for(size_t m = 0; m < oldObjModelers.size(); m++)
		if(modelIsVisible[m])
			oldObjModelers[m].lastUpdatedFrame = frameIndex; //TODO set updatedFrame here iff you want to visualize this model
	cout << "visible old models: ";
	for(size_t i = 0; i < modelIsVisible.size(); i++)
		if(modelIsVisible[i])
			cout << i << ' ';
	cout << endl;

	/*
	 * create or update a new old-obj model for each large conn comp of moved pts
	 */

	std::vector<sumLogprobsSeparate> frameSums(oldObjModelers.size());
	std::vector<std::shared_ptr<std::vector<float>>> objModelRenderedPts(oldObjModelers.size()); //for creating k-d trees later
	std::vector<std::shared_ptr<std::vector<int>>> objModelRenderingValidity(oldObjModelers.size()); //for creating k-d trees later
{
	std::shared_ptr<rgbdSensorNoiseModel> noiseModel2(new approxRGBDSensorNoiseModel(dynamic_cast<approxRGBDSensorNoiseModel&>(*noiseModel)));
	for(size_t i = 0; i < oldObjModelers.size(); i++)
		if(modelIsVisible[i])
		{
			oldObjModelerInfo& modelerInfo = oldObjModelers[i];

			/*
			 * diff model wrt cur frame
			 */

			const rgbd::eigen::Affine3f scene2ToScene1Xform = rgbd::eigen::Affine3f::Identity();

			rgbd::timer t;
			modelerInfo.modeler->render(modelerInfo.initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes[frameIndex]]);
			cv::Mat_<cv::Vec4b> renderedCols(camParams.yRes, camParams.xRes, cv::DataType<cv::Vec4b>::type); //bgra
	//		std::shared_ptr<std::vector<float>> renderedPts(new std::vector<float>(camParams.yRes * camParams.xRes * 4));
			std::shared_ptr<std::vector<float>>& renderedPts = objModelRenderedPts[i];
			renderedPts.reset(new std::vector<float>(camParams.yRes * camParams.xRes * 4));
			std::shared_ptr<std::vector<float>> renderedNormals(new std::vector<float>(camParams.yRes * camParams.xRes * 4));
	//		std::shared_ptr<std::vector<int>> renderingValidity(new std::vector<int>(camParams.yRes * camParams.xRes)); //iff 0 at a pixel, pts, normals and maybe colors are nan
			std::shared_ptr<std::vector<int>>& renderingValidity = objModelRenderingValidity[i];
			renderingValidity.reset(new std::vector<int>(camParams.yRes * camParams.xRes)); //iff 0 at a pixel, pts, normals and maybe colors are nan
			modelerInfo.modeler->getLastRender(renderedCols, *renderedPts, *renderedNormals, *renderingValidity);

			std::shared_ptr<sceneSampler> scene1sampler(new sceneSamplerRenderedTSDF(camParams, renderedPts, renderedCols, renderedNormals, renderingValidity));
			const projectSamplesIntoCameraPerPixelFunc sampleProjectionFunc = [&renderedPts,&renderingValidity](const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camXform, boost::multi_array<uint32_t, 2>& sampleIDs, boost::multi_array<float, 2>& sampleDepths, cv::Mat_<float>& sampleDepthStdevs, bool& stdevsSet)
				{
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
					{
						for(size_t j = 0; j < camParams.xRes; j++, l++)
						{
							sampleIDs[i][j] = (*renderingValidity)[l] ? (l + 1) : 0; //0 in this map is a flag for no sample
							sampleDepths[i][j] = (*renderingValidity)[l] ? (*renderedPts)[l * 4 + 2] : -1;
						}
					}

					const cv::Mat_<float> depthsMat(camParams.yRes, camParams.xRes, sampleDepths.data()); //won't try to deallocate when dies
					boost::multi_array<float, 2> sampleDepthStdevsMA(boost::extents[camParams.yRes][camParams.xRes]);
					computeDepthMapStdevsPointwise(depthsMat, sampleDepthStdevsMA, primesensor::stereoError(1), 0, true/* multithread */);
					//computeDepthMapLocalStdevs(depthsMat, 4/* nbrhood */, sampleDepthStdevsMA, 0, true/* multithread */);
					std::copy(sampleDepthStdevsMA.data(), sampleDepthStdevsMA.data() + sampleDepthStdevsMA.num_elements(), reinterpret_cast<float*>(sampleDepthStdevs.data));
					stdevsSet = true;
				};
			t.stop("set up per-frame diffing data");
			t.restart();
			noiseModel2->cachePerPointInfo(*scene1sampler);
			t.stop("cache diffing per-point info");

			t.restart();
			sceneDifferencer differ(cams);
			differ.setPrevScene(scene1sampler);
			differ.setCurFrame(scene2info, scene2ToScene1Xform);
			t.stop("set up diffing frame");
			t.restart();
			differ.projectSceneIntoFrame(sampleProjectionFunc);
			t.stop("project");
			t.restart();
			singleFrameDifferencingParams diffingParams;

			/*
			 * diff wrt new frame
			 */
			frameSums[i].init(camParams.yRes * camParams.xRes);
			diffingParams.differenceFrameWrtCloud = true;
			diffingParams.outputWrtCloud = false;
			differ.runDifferencingAfterProjection(*noiseModel2, frameSums[i], diffingParams);
			frameSums[i].weightComponents(posWeight, colWeight, normalWeight);
	#if 1
			const cv::Mat_<cv::Vec3b> diffImg = visualizeLogProbsWrtFrame(camParams, frameSums[i].logprobsDepthGivenMoved, frameSums[i].logprobsDepthGivenNotMoved, true/* multithread */);
			cv::imwrite((outdir / (boost::format("diffOldModel%1%frame%2%.png") % i % frameIndex).str()).string(), diffImg);
	#endif

#ifndef OLD_MODELS_FROM_FRAMES
			/*
			 * carve out space that's not part of the obj in the model (useful when we create the model by copying the bkgnd map)
			 *
			 * TODO we really want to only carve out space up to other surfaces, but not include those surfaces
			 */
			const rgbd::eigen::VectorXd movedProbs = frameSums[i].movedProbs();
			cv::Mat_<float> maskedDepth(camParams.yRes, camParams.xRes, 0.0f); //could use prevRenderedDepth.clone() but its quality is very low and it doesn't look good when used
#if 0 //experiment 20130914 trying to avoid destroying the object through carving -- TODO does work better, but is it nec?
			cv::Mat_<uint8_t> movedMask(camParams.yRes, camParams.xRes);
			for(size_t i = 0, l = 0; i < camParams.yRes; i++)
				for(size_t j = 0; j < camParams.xRes; j++, l++)
					if(movedProbs[l] < .5/*probNonmovedThreshold*/) movedMask(i, j) = 255;
					else movedMask(i, j) = 0;
			cv::Mat_<float> distToMoved;
			cv::distanceTransform(movedMask, distToMoved, CV_DIST_L1, CV_DIST_MASK_PRECISE);
			for(size_t i = 0, l = 0; i < camParams.yRes; i++)
				for(size_t j = 0; j < camParams.xRes; j++, l++)
					if(distToMoved(i, j) > 2/* TODO ? */) //only carve at pixels not too close to likely object pixels
					{
						maskedDepth(i, j) = 100;
					}
#else
			for(size_t i = 0, l = 0; i < camParams.yRes; i++)
				for(size_t j = 0; j < camParams.xRes; j++, l++)
					if(movedProbs[l] < probNonmovedThreshold)
					{
						maskedDepth(i, j) = 100;
					}
#endif
			Frame frameToAdd(*cl_ptr);
			frameToAdd.mat_color = frame.getColorImg(); //doesn't matter -- won't be used (it would be if the modeler were a grid map, and the map would also get huge--something to be wary of)
			frameToAdd.mat_depth = maskedDepth;
			frameToAdd.mat_segments = cv::Mat(frameToAdd.mat_color.size(), CV_32S, cv::Scalar(0));
			modelerInfo.modeler->addFrame(frameToAdd, modelerInfo.initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes[frameIndex]]);
#endif
		}
}

#define USE_KDTREES_TO_ASSOCIATE
#ifdef USE_KDTREES_TO_ASSOCIATE
	std::vector<boost::shared_ptr<kdtree2>> objModelCloudTrees(oldObjModelersSizeBeforeCreating); //for each model, a tree of all pts that appear on screen (we really want all model pts, but this is much easier to get)
	for(size_t m = 0; m < oldObjModelersSizeBeforeCreating; m++)
		if(modelIsVisible[m])
		{
			std::vector<rgbd::eigen::Vector3f> pts3d;
			for(size_t i = 0, l = 0; i < camParams.yRes; i++)
				for(size_t j = 0; j < camParams.xRes; j++, l++)
					if((*objModelRenderingValidity[m])[l])
					{
						const float x = (*objModelRenderedPts[m])[l * 4 + 0], y = (*objModelRenderedPts[m])[l * 4 + 1], z = (*objModelRenderedPts[m])[l * 4 + 2];
						pts3d.push_back(rgbd::eigen::Vector3f(x, y, z));
					}
			if(!pts3d.empty()) objModelCloudTrees[m] = rgbd::createKDTree2(pts3d);
		}

	//free memory
	objModelRenderedPts.clear();
	objModelRenderingValidity.clear();
#endif

	/*
	 * keep track of which models should be merged after updating
	 */
	std::vector<size_t> rankMap(oldObjModelersSizeBeforeCreating), parentMap(oldObjModelersSizeBeforeCreating);
	boost::iterator_property_map<std::vector<size_t>::iterator, boost::typed_identity_property_map<size_t>> rankMapAdaptor(rankMap.begin()), parentMapAdaptor(parentMap.begin());
	boost::disjoint_sets<decltype(rankMapAdaptor), decltype(parentMapAdaptor)> modelMergingSets(rankMapAdaptor, parentMapAdaptor);
	for(size_t i = 0; i < oldObjModelersSizeBeforeCreating; i++)
		modelMergingSets.make_set(i);

	for(const auto& c : compsByRepOld)
	{
		if(c.second.size() > minObjCompSizeInPixels //if the segment is reasonably sized
			&& movedProbsOld[c.first] > probMovedThreshold) //if the segment is of high-p(m) pixels
		{
			//ensure the segment is reasonably shaped, unlikely to be noise at the edge of a non-moved object
			float edgePixelPct = 0;
			for(size_t l : c.second)
			{
				const size_t i = l / camParams.xRes, j = l % camParams.xRes;
				if((i > 0 && movedProbsOld[l - camParams.xRes] < probMovedThreshold) || (i < camParams.yRes - 1 && movedProbsOld[l + camParams.xRes] < probMovedThreshold)
					|| (j > 0 && movedProbsOld[l - 1] < probMovedThreshold) || (j < camParams.xRes && movedProbsOld[l + 1] < probMovedThreshold))
					edgePixelPct += 1;
			}
			edgePixelPct /= c.second.size();
			cout << "seg " << c.first << ": size " << c.second.size() << ", edge pct " << edgePixelPct << endl;
			if(edgePixelPct < edgePixelPctThreshold)
			{
				cout << "comp " << c.first << " of size " << c.second.size() << endl;

				std::vector<size_t> modelIndicesToUpdate;
#ifdef USE_KDTREES_TO_ASSOCIATE
				/*
				 * use 3-d distance to choose existing models to add the comp to
				 *
				 * we'll compute closest distance from each "edge" point of the image segment (as a heuristic) to each existing modeler's cloud
				 *
				 * TODO allow to add to multiple models; then we don't need these k-d trees
				 */
				std::vector<size_t> compEdgePts; //img seg pts w/ low-p(m) nbrs
				for(size_t l : c.second)
				{
					const uint32_t i = l / camParams.xRes, j = l % camParams.xRes;
					if((i > 0 && movedProbsOld[l - camParams.xRes] < probMovedThreshold)
						|| (i < camParams.yRes - 1 && movedProbsOld[l + camParams.xRes] < probMovedThreshold)
						|| (j > 0 && movedProbsOld[l - 1] < probMovedThreshold)
						|| (j < camParams.xRes - 1 && movedProbsOld[l + 1] < probMovedThreshold))
					{
						compEdgePts.push_back(l);
					}
				}
				for(size_t m = 0; m < oldObjModelersSizeBeforeCreating; m++)
					if(objModelCloudTrees[m]) //if the model is eligible for having comps associated to it
					{
						float minDist = FLT_MAX;
						const kdtreeNbrhoodSpec nspec = kdtreeNbrhoodSpec::byCount(1);
						for(size_t l : compEdgePts)
						{
							const uint32_t i = l / camParams.xRes, j = l % camParams.xRes;
							const float z = renderedBkgndDepth(i, j), x = (float)(j - camParams.centerX) / camParams.focalLength * z, y = (float)(i - camParams.centerY) / camParams.focalLength * z;
							const std::vector<float> qpt = {x, y, z};
							const std::vector<kdtree2_result> nbrs = searchKDTree(*objModelCloudTrees[m], nspec, qpt);
							ASSERT_ALWAYS(nbrs.size() == 1);
							if(nbrs[0].dis < minDist) minDist = nbrs[0].dis;
							if(nbrs[0].dis < sqr(maxOldObjSameCompDist))
							{
								modelIndicesToUpdate.push_back(m);
								break;
							}
						}
						cout << "model " << m << " min dist " << sqrt(minDist) << endl;
					}
#else //use diffing to associate segments w/ models
				broken b/c the pieces already in the model will have been removed from the bkgnd map by now -- TODO ??

				/*
				 * diff to choose which modeler(s) to add the segment to
				 */
				for(size_t i = 0; i < oldObjModelersSizeBeforeCreating; i++)
					if(modelIsVisible[i])
					{
						const rgbd::eigen::VectorXd movedProbsMap = frameSums[i].movedProbs(), evidenceMagnitude = frameSums[i].evidenceMagnitude();
						float score = 0;
						size_t count = 0;
						for(size_t l : c.second)
							if(evidenceMagnitude[l] > 0)
							{
								/*
								penalize moved model pts
								don't penalize hidden model pts
								don't penalize moved scene pts
								penalize hidden scene pts
								reward nonmoved pts
								*/
								score += .5 - movedProbsMap[l];
								count++;
							}
						score = (count == 0) ? 0 : (score / count);
						cout << "old model " << i << ", comp " << c.first << ": diff score " << score << endl;
						if(score > .4/* TODO ?; max is .5 */) modelIndicesToUpdate.push_back(i);
					}
#endif

				if(modelIndicesToUpdate.empty()) //if the seg doesn't belong to any existing obj modeler
				{
					cout << "creating old modeler #" << oldObjModelers.size() << " for seg " << c.first << endl;

					/*
					 * compute the bbox of the new modeler in current camera coords
					 *
					 * can't go farther in x-y than bounds of the seg + all connected unseen pixels (but this affects the z-limits; for now just add a constant to the max z of the segment?)
						can't go closer than closest seg pt dist
						can't go farther than farthest seg pt dist in new frame (unless there are unseen pixels, in which case need an arbitrary limit, but ignore that for now)
					 */
					rgbd::eigen::Vector3f mins, maxes; //in the camera's frame
				{
					/*
					 * find all unseen and occluded pts connected to the comp in 2-d
					 */

					std::unordered_set<size_t> compPixelsSet(c.second.begin(), c.second.end()); //all pixels in the comp

					std::vector<size_t> rankMap(camParams.yRes * camParams.xRes), parentMap(camParams.yRes * camParams.xRes);
					boost::iterator_property_map<std::vector<size_t>::iterator, boost::typed_identity_property_map<size_t>> rankMapAdaptor(rankMap.begin()), parentMapAdaptor(parentMap.begin());
					boost::disjoint_sets<decltype(rankMapAdaptor), decltype(parentMapAdaptor)> sets(rankMapAdaptor, parentMapAdaptor);
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
						for(size_t j = 0; j < camParams.xRes; j++, l++)
							sets.make_set(l);
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
						for(size_t j = 0; j < camParams.xRes; j++, l++)
						{
							if(j < camParams.xRes - 1 && (compPixelsSet.find(l) != compPixelsSet.end() || movedProbsOld[l] == .5) && (compPixelsSet.find(l + 1) != compPixelsSet.end() || movedProbsOld[l + 1] == .5))
								sets.union_set(l, l + 1);
							if(i < camParams.yRes - 1 && (compPixelsSet.find(l) != compPixelsSet.end() || movedProbsOld[l] == .5) && (compPixelsSet.find(l + camParams.xRes) != compPixelsSet.end() || movedProbsOld[l + camParams.xRes] == .5))
								sets.union_set(l, l + camParams.xRes);
						}
					std::vector<size_t> umbraPts; //pts in the conn comp + close-by unseen pts
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
						for(size_t j = 0; j < camParams.xRes; j++, l++)
							if(sets.find_set(l) == sets.find_set(c.second[0]))
								umbraPts.push_back(l);
					cout << "comp size " << c.second.size() << ", umbra size " << umbraPts.size() << endl;
					ASSERT_ALWAYS(umbraPts.size() >= c.second.size()); //the umbra includes the conn comp

					uint32_t minU = camParams.xRes - 1, maxU = 0, minV = camParams.yRes - 1, maxV = 0; //in pixels
					for(size_t l : umbraPts)
					{
						const uint32_t v = l / camParams.xRes, u = l % camParams.xRes;
						if(u < minU) minU = u;
						if(u > maxU) maxU = u;
						if(v < minV) minV = v;
						if(v > maxV) maxV = v;
					}
					float minZ = FLT_MAX, maxZ = -FLT_MAX; //in m
					for(size_t l : c.second)
					{
						const uint32_t i = l / camParams.xRes, j = l % camParams.xRes;
						const float zBkgnd = renderedBkgndDepth(i, j), zFrame = depth(i, j);
						if(zBkgnd > 0 && zBkgnd < minZ) minZ = zBkgnd;
						if(zFrame > maxZ) maxZ = zFrame;
					}
					minZ -= .15; //to account for parts of the obj not seen in this first segment; TODO not necessary once we resize models during frame add
					maxZ += .15; //to account for unseen pixels next to the segment; TODO not necessary once we can resize single-volume models during merge
					//find x-y bounds big enough for the pixels and z-values we want -- whether this happens at min or max z depends on sign of (minU - cx) etc
					const float minX = std::min(((float)minU - camParams.centerX) * minZ, ((float)minU - camParams.centerX) * maxZ) / camParams.focalLength,
						maxX = std::max(((float)maxU - camParams.centerX) * minZ, ((float)maxU - camParams.centerX) * maxZ) / camParams.focalLength,
						minY = std::min(((float)minV - camParams.centerY) * minZ, ((float)minV - camParams.centerY) * maxZ) / camParams.focalLength,
						maxY = std::max(((float)maxV - camParams.centerY) * minZ, ((float)maxV - camParams.centerY) * maxZ) / camParams.focalLength;
					mins = rgbd::eigen::Vector3f(minX, minY, minZ);
					maxes = rgbd::eigen::Vector3f(maxX, maxY, maxZ);
				}
					cout << "map3d mins " << mins.transpose() << ", maxes " << maxes.transpose() << endl;

					/*
					 * create volume
					 */
					VolumeModelerAllParams modelParams;
					modelParams.volume_modeler.model_type = MODEL_SINGLE_VOLUME; //TODO ?

					float modelerVolume = 1;
					for(size_t k = 0; k < 3; k++) modelerVolume *= maxes[k] - mins[k];
					const size_t maxModelerCells = pow(192, 3); //TODO ?
					const float minCellSize = pow(modelerVolume / maxModelerCells, 1 / 3.0); //limit resolution by vram
					modelParams.volume.cell_size = std::max(.002f/* TODO ? */, minCellSize); //avoid absurdly small resolution in any case
					cout << " old model has resolution " << modelParams.volume.cell_size << endl;
					for(size_t k = 0; k < 3; k++) modelParams.volume.cell_count[k] = (int)ceil((maxes[k] - mins[k]) / modelParams.volume.cell_size);
					cout << " old model #cells: " << modelParams.volume.cell_count.transpose() << endl;

					modelParams.camera = prevSceneMapParams.camera;
					std::shared_ptr<VolumeModeler> modeler(new VolumeModeler(clKernels, modelParams));

					oldObjModelerInfo modelerInfo;
					modelerInfo.modeler = modeler;
					modelerInfo.params = modelParams;
					modelerInfo.initialFrame = frameIndex;
					modelerInfo.initialCamPoseWrtMap = rgbd::eigen::Translation3f(-mins);
					modelerInfo.lastUpdatedFrame = modelerInfo.initialFrame;

#ifdef OLD_MODELS_FROM_FRAMES //for the icra deadline, use a frame instead of the bkgnd map so models look nicer; then TODO figure out how to make the map-copying + space carving work better
					/*
					 * make a frame with just the segment
					 */
					cv::Mat_<float> maskedDepth(camParams.yRes, camParams.xRes, 0.0f); //do we want to do space carving here? -- 20130914 no, it looks awful with carving
					for(size_t l : c.second)
					{
						const size_t i = l / camParams.xRes, j = l % camParams.xRes;
						maskedDepth(i, j) = renderedBkgndDepth(i, j);
					}
					Frame frameToAdd(*cl_ptr);
					frameToAdd.mat_color = renderedBkgndCols;
					frameToAdd.mat_depth = maskedDepth;
					frameToAdd.setColorWeights(4, 4); //(10, 5) suggested by peter and does improve colors; I get better-shaped models with (4, 4)
#if 0
					rgbd::writeDepthMapValuesImg(maskedDepth, outdir / (boost::format("addsegOld_f%1%_s%2%.png") % frameIndex % c.first).str());
#endif

					modeler->addFrame(frameToAdd, modelerInfo.initialCamPoseWrtMap);
#else
					/*
					 * fill the volume with part of the bkgnd map
					 */
					modeler->mergeOtherVolume(*bkgndMapper, mapPosesWrtGlobal[scene1.sceneName].inverse() * mapPosesWrtGlobal[scene2.sceneName] * newFramePosesWrtCurMap[frameTimes[frameIndex]] * modelerInfo.initialCamPoseWrtMap.inverse());
					//modeler->setMaxWeightInVolume(3.0/* TODO ? */); //allow space carving to happen
#endif

					//TODO carve out space in the model that's not part of the obj from this viewpoint; if we don't we might get things like a door and a chair in front of the door both being in the old but not the new map,
					//and the door getting included in both old models and not getting removed from the chair model due to no longer being in the bkgnd map
					//(for this purpose we'll also need to set the max weight in these old models to be low before doing this carving)

					oldObjModelers.push_back(modelerInfo);
				}
				else
				{
					/*
					 * make a frame with just the segment
					 */
					cv::Mat_<float> maskedDepth(camParams.yRes, camParams.xRes, 0.0f); //do we want to do space carving here (use a default z other than zero)? -- 20130914 no, it looks awful with carving
					for(size_t l : c.second)
					{
						const int32_t i = l / camParams.xRes, j = l % camParams.xRes;
#if 0 //add pixels in the vicinity even though they might not be part of the obj -- verdict: doesn't noticeably improve models
						const int32_t hw = 1; //TODO ?
						for(int32_t ii = std::max((int32_t)0, i - hw); ii <= std::min((int32_t)camParams.yRes - 1, i + hw); ii++)
							for(int32_t jj = std::max((int32_t)0, j - hw); jj <= std::min((int32_t)camParams.xRes - 1, j + hw); jj++)
								maskedDepth(ii, jj) = renderedBkgndDepth(ii, jj);
#else
						maskedDepth(i, j) = renderedBkgndDepth(i, j);
#endif
					}
					Frame frameToAdd(*cl_ptr);
					frameToAdd.mat_color = renderedBkgndCols;
					frameToAdd.mat_depth = maskedDepth;
					frameToAdd.setColorWeights(4, 4); //(10, 5) suggested by peter and does improve colors; I get better-shaped models with (4, 4)

					for(size_t i = 1; i < modelIndicesToUpdate.size(); i++) modelMergingSets.union_set(modelIndicesToUpdate[0], modelIndicesToUpdate[i]);
					for(size_t modelIndex : modelIndicesToUpdate)
					{
						cout << "adding seg " << c.first << " to old modeler #" << modelIndex << endl;
						//TODO resize the modeler if the new seg extends outside it
						oldObjModelerInfo& modelerInfo = oldObjModelers[modelIndex];
						frameToAdd.setColorWeights(4, 4); //(10, 5) suggested by peter and does improve colors; I get better-shaped models with (4, 4)
						modelerInfo.modeler->addFrame(frameToAdd, modelerInfo.initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes[frameIndex]]);
						modelerInfo.lastUpdatedFrame = frameIndex;
					}

					cout << "model merging sets now:" << endl;
					for(size_t i = 0; i < oldObjModelersSizeBeforeCreating; i++)
						cout << " " << i << " in " << modelMergingSets.find_set(i) << endl;
				}
			}
		}
	}

	/*
	 * merge models
	 */
	std::unordered_map<size_t, std::vector<size_t>> modelMergingCompsByRep;
	for(size_t i = 0; i < oldObjModelersSizeBeforeCreating; i++) modelMergingCompsByRep[modelMergingSets.find_set(i)].push_back(i);
	std::vector<size_t> modelIndicesToRemove;
	std::vector<oldObjModelerInfo> modelsToAdd;
	for(const auto& c : modelMergingCompsByRep)
		if(c.second.size() > 1)
		{
			cout << "merging old-obj models "; std::copy(c.second.begin(), c.second.end(), std::ostream_iterator<size_t>(cout, " ")); cout << endl;

			/*
			 * create a new map large enough to hold all existing ones
			 *
			 * put it in the whole-scene volumetric map's coord frame, for simplicity
			 */
			rgbd::eigen::Vector3f mins(FLT_MAX, FLT_MAX, FLT_MAX), maxes(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			for(size_t i = 0; i < c.second.size(); i++)
			{
				const oldObjModelerInfo& modelerInfo = oldObjModelers[c.second[i]];
				const auto& volumeParams = modelerInfo.params.volume;
				const rgbd::eigen::Affine3f modelerMapPoseWrtSceneMap = newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]] * modelerInfo.initialCamPoseWrtMap.inverse();
				rgbd::eigen::Vector3f modelMins(FLT_MAX, FLT_MAX, FLT_MAX), modelMaxes(-FLT_MAX, -FLT_MAX, -FLT_MAX); //for debugging; TODO remove
				for(size_t ix = 0; ix < 2; ix++)
				{
					for(size_t iy = 0; iy < 2; iy++)
					{
						for(size_t iz = 0; iz < 2; iz++)
						{
							const rgbd::eigen::Vector3f corner = modelerMapPoseWrtSceneMap
								* (rgbd::eigen::Vector3f(ix * volumeParams.cell_count[0], iy * volumeParams.cell_count[1], iz * volumeParams.cell_count[2]) * volumeParams.cell_size);
							for(size_t k = 0; k < 3; k++)
							{
								if(corner[k] < mins[k]) mins[k] = corner[k];
								if(corner[k] > maxes[k]) maxes[k] = corner[k];

								if(corner[k] < modelMins[k]) modelMins[k] = corner[k];
								if(corner[k] > modelMaxes[k]) modelMaxes[k] = corner[k];
							}
						}
					}
				}
				cout << " old model " << c.second[i] << ": mins " << modelMins.transpose() << ", maxes " << modelMaxes.transpose() << endl;
				cout << "  volume size: " << (modelerInfo.params.volume.cell_count.cast<float>() * modelerInfo.params.volume.cell_size).transpose() << endl;
			}

			VolumeModelerAllParams modelParams;
			modelParams.volume_modeler.model_type = MODEL_SINGLE_VOLUME; //TODO ?

			float modelerVolume = 1;
			for(size_t k = 0; k < 3; k++) modelerVolume *= maxes[k] - mins[k];
			const size_t maxModelerCells = pow(192, 3); //TODO ?
			const float minCellSize = pow(modelerVolume / maxModelerCells, 1 / 3.0); //limit resolution by vram
			modelParams.volume.cell_size = std::max(.002f/* TODO ? */, minCellSize); //avoid absurdly small resolution in any case
			cout << " merged old model has resolution " << modelParams.volume.cell_size << endl;
			for(size_t k = 0; k < 3; k++) modelParams.volume.cell_count[k] = (int)ceil((maxes[k] - mins[k]) / modelParams.volume.cell_size);
			cout << " merged old model #cells: " << modelParams.volume.cell_count.transpose() << endl;

			modelParams.volume_modeler.first_frame_centroid = false; //center the volume on the centroid of the valid pixels in the first frame given to it
			modelParams.camera = prevSceneMapParams.camera;
			std::shared_ptr<VolumeModeler> modeler(new VolumeModeler(clKernels, modelParams));

			oldObjModelerInfo modelerInfo;
			modelerInfo.modeler = modeler;
			modelerInfo.params = modelParams;
			modelerInfo.initialFrame = 0;
			modelerInfo.initialCamPoseWrtMap = rgbd::eigen::Translation3f(-mins);
			modelerInfo.lastUpdatedFrame = frameIndex;
			cout << " mins " << mins.transpose() << ", maxes " << maxes.transpose() << endl;

			for(size_t i = 0; i < c.second.size(); i++)
			{
				//TODO downweight confidence when merging, or maybe when copying the bkgnd map into old models to start with, to make things overwritable?
				cout << " merging in model " << c.second[i] << endl;
				modeler->mergeOtherVolume(*oldObjModelers[c.second[i]].modeler, oldObjModelers[c.second[i]].initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[oldObjModelers[c.second[i]].initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]] * modelerInfo.initialCamPoseWrtMap.inverse());
#if 0 //visualize
				cv::Mat_<cv::Vec3b> mim = getPrettyIntelMapRender(*modeler, camParams, modelerInfo.initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes.back()]);
				cv::imwrite((boost::format("mimold%2%_%1%.png") % i % frameIndex).str(), mim);
#endif
				oldObjModelers[c.second[i]].modeler.reset(); //save gpu ram
				modelIndicesToRemove.push_back(c.second[i]);
			}
			modelsToAdd.push_back(modelerInfo);
		}
	std::sort(modelIndicesToRemove.begin(), modelIndicesToRemove.end()); //required for removeFromRange()
	oldObjModelers.erase(removeFromRange(oldObjModelers, modelIndicesToRemove), oldObjModelers.end());
	for(size_t i = 0; i < modelsToAdd.size(); i++) cout << "adding merged old model " << (oldObjModelers.size() + i) << endl;
	oldObjModelers.insert(oldObjModelers.end(), modelsToAdd.begin(), modelsToAdd.end());

	/*
	 * push models that haven't been updated recently off the gpu (they'll be automatically pulled back in when needed)
	 */
	size_t numModelsOnGPU = 0;
	for(oldObjModelerInfo& modelerInfo : oldObjModelers)
		if(modelerInfo.lastUpdatedFrame + modelInactivityDeallocationPeriod < frameIndex)
			modelerInfo.modeler->deallocateBuffers();
		else
			numModelsOnGPU++;
	cout << numModelsOnGPU << " old obj models are in vram" << endl;

	t.stop("update and create old-obj modelers");
}

	/*************************************************************************************************************************************************************************************************************************************************
	 * create and update models of objects that appear in the new scene and not in the old
	 */
{
	t.restart();

	/*
	 * find conn comps of high-p(m) points in the new frame
	 */
	const rgbd::eigen::VectorXd movedProbsNew = lastFrameLogprobSumsWrtMap.movedProbs();
	std::unordered_map<size_t, std::vector<size_t>> compsByRepNew = findDiffingConnComps(camParams, frame.getDepthImg(), movedProbsNew, probMovedThreshold, diffingConnCompsAlgorithm::LARGE_NBRHOOD_2D, .01);
	//TODO split comps using previously obtained segm info from existing models

	const size_t newObjModelersSizeBeforeCreating = newObjModelers.size(); //so we won't look for diffing results wrt modelers we've created during the same frame

	/*
	 * list only models whose volumes intersect the viewing frustum, so we can avoid keeping the others in gpu memory
	 */
	const std::vector<bool> modelIsVisible = listObjModelVisibility(newObjModelers, maxFrustumZ);
	for(size_t m = 0; m < newObjModelers.size(); m++)
		if(modelIsVisible[m])
			newObjModelers[m].lastUpdatedFrame = frameIndex; //TODO set updatedFrame here iff you want to visualize this model
	cout << "visible new models: ";
	for(size_t i = 0; i < modelIsVisible.size(); i++)
		if(modelIsVisible[i])
			cout << i << ' ';
	cout << endl;

	/*
	 * create or update a model for each large conn comp of moved pts
	 */

	std::vector<sumLogprobsSeparate> frameSums(newObjModelers.size());
{
	std::shared_ptr<approxRGBDSensorNoiseModel> noiseModel2(new approxRGBDSensorNoiseModel(dynamic_cast<approxRGBDSensorNoiseModel&>(*noiseModel)));
	for(size_t i = 0; i < newObjModelers.size(); i++)
		if(modelIsVisible[i])
		{
			const rgbd::eigen::Affine3f scene2ToScene1Xform = rgbd::eigen::Affine3f::Identity();

			rgbd::timer t;
			newObjModelers[i].modeler->render(newObjModelers[i].initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[newObjModelers[i].initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes[frameIndex]]);
			cv::Mat_<cv::Vec4b> renderedCols(camParams.yRes, camParams.xRes, cv::DataType<cv::Vec4b>::type); //bgra
			std::shared_ptr<std::vector<float>> renderedPts(new std::vector<float>(camParams.yRes * camParams.xRes * 4));
			std::shared_ptr<std::vector<float>> renderedNormals(new std::vector<float>(camParams.yRes * camParams.xRes * 4));
			std::shared_ptr<std::vector<int>> renderingValidity(new std::vector<int>(camParams.yRes * camParams.xRes)); //iff 0 at a pixel, pts, normals and maybe colors are nan
			newObjModelers[i].modeler->getLastRender(renderedCols, *renderedPts, *renderedNormals, *renderingValidity);

			std::shared_ptr<sceneSampler> scene1sampler(new sceneSamplerRenderedTSDF(camParams, renderedPts, renderedCols, renderedNormals, renderingValidity));
			const projectSamplesIntoCameraPerPixelFunc sampleProjectionFunc = [&renderedPts,&renderingValidity](const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camXform, boost::multi_array<uint32_t, 2>& sampleIDs, boost::multi_array<float, 2>& sampleDepths, cv::Mat_<float>& sampleDepthStdevs, bool& stdevsSet)
				{
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
					{
						for(size_t j = 0; j < camParams.xRes; j++, l++)
						{
							sampleIDs[i][j] = (*renderingValidity)[l] ? (l + 1) : 0; //0 in this map is a flag for no sample
							sampleDepths[i][j] = (*renderingValidity)[l] ? (*renderedPts)[l * 4 + 2] : -1;
						}
					}

					const cv::Mat_<float> depthsMat(camParams.yRes, camParams.xRes, sampleDepths.data()); //won't try to deallocate when dies
					boost::multi_array<float, 2> sampleDepthStdevsMA(boost::extents[camParams.yRes][camParams.xRes]);
					computeDepthMapLocalStdevs(depthsMat, 4/* nbrhood */, sampleDepthStdevsMA, 0, camParams, true/* multithread */);
#if 1
					boost::multi_array<float, 2> sampleDepthStdevsMA2(boost::extents[camParams.yRes][camParams.xRes]);
					computeDepthMapStdevsPointwise(depthsMat, sampleDepthStdevsMA2, primesensor::stereoError(1), 0, true/* multithread */);
					for(size_t i = 0, l = 0; i < camParams.yRes; i++)
						for(size_t j = 0; j < camParams.xRes; j++, l++)
							*(sampleDepthStdevsMA.data() + l) = std::max(*(sampleDepthStdevsMA.data() + l), *(sampleDepthStdevsMA2.data() + l));
#endif
					std::copy(sampleDepthStdevsMA.data(), sampleDepthStdevsMA.data() + sampleDepthStdevsMA.num_elements(), reinterpret_cast<float*>(sampleDepthStdevs.data));
					stdevsSet = true;
				};
			t.stop("set up per-frame diffing data");
			t.restart();
			noiseModel2->cachePerPointInfo(*scene1sampler);
			t.stop("cache diffing per-point info");

			t.restart();
			sceneDifferencer differ(cams);
			differ.setPrevScene(scene1sampler);
			differ.setCurFrame(scene2info, scene2ToScene1Xform);
			t.stop("set up diffing frame");
			t.restart();
			differ.projectSceneIntoFrame(sampleProjectionFunc);
			t.stop("project");
			t.restart();
			singleFrameDifferencingParams diffingParams;

#if 1 //debugging
			cv::Mat_<float> dzMap(camParams.yRes, camParams.xRes);
			for(size_t i = 0, l = 0; i < camParams.yRes; i++)
				for(size_t j = 0; j < camParams.xRes; j++, l++)
				{
					const float renderedZ = (*renderingValidity)[l] ? (*renderedPts)[l * 4 + 2] : -1;
					const float obsZ = frame.getDepthImg()(i, j);
					dzMap(i, j) = fabs(obsZ - renderedZ);
				}
			rgbd::writeDepthMapValuesImg(dzMap, outdir / (boost::format("diffModelDZ%1%_m%2%.png") % frameIndex % i).str());
#endif

			/*
			 * diff wrt new frame
			 */
//			noiseModel2->setSigmaD0(newObjModelers[i].params.volume.cell_size/* TODO cell size multiplier? */); ?
			frameSums[i].init(camParams.yRes * camParams.xRes);
			diffingParams.differenceFrameWrtCloud = true;
			diffingParams.outputWrtCloud = false;
			differ.runDifferencingAfterProjection(*noiseModel2, frameSums[i], diffingParams);
			frameSums[i].weightComponents(posWeight, colWeight, normalWeight);
	#if 0
			cout << "frame " << frameIndex << ", new model " << i << ": cell size " << newObjModelers[i].params.volume.cell_size << endl;
			cv::Mat_<cv::Vec3b> bgrImg(renderedCols.size());
			for(int i = 0; i < 480; i++)
				for(int j = 0; j < 640; j++)
				{
					const cv::Vec4b c = renderedCols(i, j);
					bgrImg(i, j) = cv::Vec3b(c[0], c[1], c[2]);
				}
			cv::imwrite((outdir / (boost::format("renderedNewModel%1%frame%2%.png") % i % frameIndex).str()).string(), bgrImg);
			const cv::Mat_<cv::Vec3b> diffImg = visualizeLogProbsWrtFrame(camParams, frameSums[i].logprobsDepthGivenMoved, frameSums[i].logprobsDepthGivenNotMoved, true/* multithread */);
			cv::imwrite((outdir / (boost::format("diffNewModel%1%frame%2%.png") % i % frameIndex).str()).string(), diffImg);
	#endif
		}
}

	/*
	 * keep track of which models should be merged after updating
	 */
	std::vector<size_t> rankMap(newObjModelersSizeBeforeCreating), parentMap(newObjModelersSizeBeforeCreating);
	boost::iterator_property_map<std::vector<size_t>::iterator, boost::typed_identity_property_map<size_t>> rankMapAdaptor(rankMap.begin()), parentMapAdaptor(parentMap.begin());
	boost::disjoint_sets<decltype(rankMapAdaptor), decltype(parentMapAdaptor)> modelMergingSets(rankMapAdaptor, parentMapAdaptor);
	for(size_t i = 0; i < newObjModelersSizeBeforeCreating; i++)
		modelMergingSets.make_set(i);

	for(const auto& c : compsByRepNew)
	{
		if(c.second.size() > minObjCompSizeInPixels //if the segment is reasonably sized
			&& movedProbsNew[c.first] > probMovedThreshold) //if the segment is of high-p(m) pixels
		{
			//ensure the segment is reasonably shaped, unlikely to be noise at the edge of a non-moved object
			float edgePixelPct = 0;
			for(size_t l : c.second)
			{
				const size_t i = l / camParams.xRes, j = l % camParams.xRes;
				if((i > 0 && movedProbsNew[l - camParams.xRes] < probMovedThreshold) || (i < camParams.yRes - 1 && movedProbsNew[l + camParams.xRes] < probMovedThreshold)
					|| (j > 0 && movedProbsNew[l - 1] < probMovedThreshold) || (j < camParams.xRes && movedProbsNew[l + 1] < probMovedThreshold))
					edgePixelPct += 1;
			}
			edgePixelPct /= c.second.size();
			cout << "seg " << c.first << ": size " << c.second.size() << ", edge pct " << edgePixelPct << endl;
			if(edgePixelPct < edgePixelPctThreshold)
			{
				cout << "comp " << c.first << " of size " << c.second.size() << endl;

				//TODO include surrounding occluded pixels in the model

				/*
				 * make a frame with just the segment
				 */
				Frame frameToAdd(*cl_ptr);
			{
				cv::Mat_<float> maskedDepth(depth.size(), 0);
				for(size_t l : c.second)
				{
					const size_t i = l / camParams.xRes, j = l % camParams.xRes;
					maskedDepth(i, j) = depth(i, j);
				}
#if 1
				rgbd::writeDepthMapValuesImg(maskedDepth, outdir / (boost::format("addsegNew_f%1%_s%2%.png") % frameIndex % c.first).str());
#endif
				frameToAdd.mat_color = frame.getColorImg();
				frameToAdd.mat_depth = maskedDepth;
				frameToAdd.setColorWeights(4, 4); //(10, 5) suggested by peter and does improve colors; I get better-shaped models with (4, 4)
			}

				std::vector<size_t> modelIndicesToUpdate;
				/*
				 * diff to choose which modeler(s) to add the segment to
				 */
				for(size_t i = 0; i < newObjModelersSizeBeforeCreating; i++)
					if(modelIsVisible[i])
					{
						const rgbd::eigen::VectorXd movedProbsMap = frameSums[i].movedProbs(), evidenceMagnitude = frameSums[i].evidenceMagnitude();
						float score = 0;
						size_t count = 0;
						for(size_t l : c.second)
							if(evidenceMagnitude[l] > 0)
							{
								/*
								penalize moved model pts
								don't penalize hidden model pts
								don't penalize moved scene pts
								penalize hidden scene pts
								reward nonmoved pts
								*/
								score += .5 - movedProbsMap[l];
								count++;
							}
						score = (count == 0) ? 0 : (score / count);
						cout << "seg " << c.first << ", new model " << i << ": diffing score " << score << endl;
						if(score > .2/* TODO ?; max is .5; depends heavily on the diffing model in use; was .4 until 20140121; .38 for obj5before/after; maybe .33 for objsmove2 (no prev scene) */) modelIndicesToUpdate.push_back(i);

#if 1 //debugging (red means the seg matches the model well)
						cv::Mat_<cv::Vec3b> dimg(480, 640);
						for(int q = 0; q < 480; q++)
							for(int r = 0; r < 640; r++)
								dimg(q, r) = cv::Vec3b(0, 0, 0);
						for(size_t l : c.second)
							if(evidenceMagnitude[l] > 0)
								dimg(l / 640, l % 640) = cv::Vec3b(0, 255 * movedProbsMap[l], 255);
						cv::imwrite((outdir / (boost::format("segModelDiff_f%1%_s%2%_m%3%.png") % frameIndex % c.first % i).str()).string(), dimg);
#endif
					}

				if(modelIndicesToUpdate.empty()) //if the seg doesn't belong to any existing obj modeler
				{
					cout << "creating new modeler #" << newObjModelers.size() << " for seg " << c.first << endl;

					VolumeModelerAllParams modelParams;
					modelParams.volume_modeler.model_type = MODEL_SINGLE_VOLUME;

					rgbd::eigen::Vector3f maxes(-FLT_MAX, -FLT_MAX, -FLT_MAX), mins(FLT_MAX, FLT_MAX, FLT_MAX);
					for(size_t l : c.second)
					{
						const size_t i = l / camParams.xRes, j = l % camParams.xRes;
						const float z = depth(i, j), x = ((float)j - camParams.centerX) * z / camParams.focalLength, y = ((float)i - camParams.centerY) * z / camParams.focalLength;
						if(x < mins[0]) mins[0] = x;
						if(y < mins[1]) mins[1] = y;
						if(z < mins[2]) mins[2] = z;
						if(x > maxes[0]) maxes[0] = x;
						if(y > maxes[1]) maxes[1] = y;
						if(z > maxes[2]) maxes[2] = z;
					}
					//add a margin for future expansion of the model
					mins -= rgbd::eigen::Vector3f::Constant(.3); //TODO ?
					maxes += rgbd::eigen::Vector3f::Constant(.3);

					float modelerVolume = 1;
					for(size_t k = 0; k < 3; k++) modelerVolume *= maxes[k] - mins[k];
					const size_t maxModelerCells = pow(192, 3); //TODO ?
					const float minCellSize = pow(modelerVolume / maxModelerCells, 1 / 3.0); //limit resolution by vram
					modelParams.volume.cell_size = std::max(.002f/* TODO ? */, minCellSize); //avoid absurdly small resolution in any case
					cout << " new model has resolution " << modelParams.volume.cell_size << endl;
					for(size_t k = 0; k < 3; k++) modelParams.volume.cell_count[k] = (int)ceil((maxes[k] - mins[k]) / modelParams.volume.cell_size);
					cout << " new model #cells: " << modelParams.volume.cell_count.transpose() << endl;

					modelParams.volume_modeler.first_frame_centroid = true; //center the volume on the centroid of the valid pixels in the first frame aligned to it
					modelParams.camera = curSceneMapParams.camera;
					std::shared_ptr<VolumeModeler> modeler(new VolumeModeler(clKernels, modelParams));

#if 1 //TODO debugging 20140429
					for(int i = 0; i < 480; i++)
						for(int j = 0; j < 640; j++)
							ASSERT_ALWAYS(frameToAdd.mat_depth.at<float>(i, j) >= 0);
#endif
					rgbd::eigen::Affine3f modelerInitialCamPoseWrtMap;
					modeler->alignFrame(frameToAdd, modelerInitialCamPoseWrtMap);
					modeler->addFrame(frameToAdd, modelerInitialCamPoseWrtMap);

					newObjModelerInfo modelerInfo;
					modelerInfo.modeler = modeler;
					modelerInfo.params = modelParams;
					modelerInfo.initialFrame = frameIndex;
					modelerInfo.initialCamPoseWrtMap = modelerInitialCamPoseWrtMap;
					modelerInfo.lastUpdatedFrame = modelerInfo.initialFrame;
					cout << "  xform" << endl << modelerInfo.initialCamPoseWrtMap.matrix() << endl;
					newObjModelers.push_back(modelerInfo);
				}
				else
				{
					for(size_t i = 1; i < modelIndicesToUpdate.size(); i++) modelMergingSets.union_set(modelIndicesToUpdate[0], modelIndicesToUpdate[i]);
					for(size_t modelIndex : modelIndicesToUpdate)
					{
						cout << "adding seg " << c.first << " to new modeler #" << modelIndex << endl;
						newObjModelerInfo& modelerInfo = newObjModelers[modelIndex];
						modelerInfo.modeler->addFrame(frameToAdd, modelerInfo.initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes[frameIndex]]);
						modelerInfo.lastUpdatedFrame = frameIndex;
					}

					//TODO debugging
					cout << "model merging sets now:" << endl;
					for(size_t i = 0; i < newObjModelersSizeBeforeCreating; i++)
						cout << " " << i << " in " << modelMergingSets.find_set(i) << endl;
				}
			}
		}
	}

	/*
	 * merge models
	 */
	std::unordered_map<size_t, std::vector<size_t>> modelMergingCompsByRep;
	for(size_t i = 0; i < newObjModelersSizeBeforeCreating; i++) modelMergingCompsByRep[modelMergingSets.find_set(i)].push_back(i);
	std::vector<size_t> modelIndicesToRemove;
	std::vector<newObjModelerInfo> modelsToAdd;
	for(const auto& c : modelMergingCompsByRep)
		if(c.second.size() > 1)
		{
			cout << "merging new-obj models "; std::copy(c.second.begin(), c.second.end(), std::ostream_iterator<size_t>(cout, " ")); cout << endl;

			/*
			 * create a new map large enough to hold all existing ones
			 *
			 * put it in the whole-scene volumetric map's coord frame, for simplicity
			 */
			rgbd::eigen::Vector3f mins(FLT_MAX, FLT_MAX, FLT_MAX), maxes(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			for(size_t i = 0; i < c.second.size(); i++)
			{
				const newObjModelerInfo& modelerInfo = newObjModelers[c.second[i]];
				const auto& volumeParams = modelerInfo.params.volume;
				const rgbd::eigen::Affine3f modelerMapPoseWrtSceneMap = newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]] * modelerInfo.initialCamPoseWrtMap.inverse();
				rgbd::eigen::Vector3f modelMins(FLT_MAX, FLT_MAX, FLT_MAX), modelMaxes(-FLT_MAX, -FLT_MAX, -FLT_MAX); //for debugging; TODO remove
				for(size_t ix = 0; ix < 2; ix++)
				{
					for(size_t iy = 0; iy < 2; iy++)
					{
						for(size_t iz = 0; iz < 2; iz++)
						{
							const rgbd::eigen::Vector3f corner = modelerMapPoseWrtSceneMap
								* (rgbd::eigen::Vector3f(ix * volumeParams.cell_count[0], iy * volumeParams.cell_count[1], iz * volumeParams.cell_count[2]) * volumeParams.cell_size);
							for(size_t k = 0; k < 3; k++)
							{
								if(corner[k] < mins[k]) mins[k] = corner[k];
								if(corner[k] > maxes[k]) maxes[k] = corner[k];

								if(corner[k] < modelMins[k]) modelMins[k] = corner[k];
								if(corner[k] > modelMaxes[k]) modelMaxes[k] = corner[k];
							}
						}
					}
				}
				cout << " new model " << c.second[i] << ": mins " << modelMins.transpose() << ", maxes " << modelMaxes.transpose() << endl;
				cout << "  volume size: " << (modelerInfo.params.volume.cell_count.cast<float>() * modelerInfo.params.volume.cell_size).transpose() << endl;
			}
			VolumeModelerAllParams modelParams;
			modelParams.volume_modeler.model_type = MODEL_SINGLE_VOLUME;

			float modelerVolume = 1;
			for(size_t k = 0; k < 3; k++) modelerVolume *= maxes[k] - mins[k];
			const size_t maxModelerCells = pow(192, 3); //TODO ?
			const float minCellSize = pow(modelerVolume / maxModelerCells, 1 / 3.0); //limit resolution by vram
			modelParams.volume.cell_size = std::max(.002f/* TODO ? */, minCellSize); //avoid absurdly small resolution in any case
			cout << " new model has resolution" << modelParams.volume.cell_size << endl;
			for(size_t k = 0; k < 3; k++) modelParams.volume.cell_count[k] = (int)ceil((maxes[k] - mins[k]) / modelParams.volume.cell_size);

			modelParams.volume_modeler.first_frame_centroid = false; //whether to center the volume on the centroid of the valid pixels in the first frame aligned to it
			modelParams.camera = curSceneMapParams.camera;
			std::shared_ptr<VolumeModeler> modeler(new VolumeModeler(clKernels, modelParams));

			newObjModelerInfo modelerInfo;
			modelerInfo.modeler = modeler;
			modelerInfo.params = modelParams;
			modelerInfo.initialFrame = 0;
			modelerInfo.initialCamPoseWrtMap = rgbd::eigen::Translation3f(-mins) * newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]];
			modelerInfo.lastUpdatedFrame = frameIndex;
			cout << " mins " << mins.transpose() << ", maxes " << maxes.transpose() << endl;

			for(size_t i = 0; i < c.second.size(); i++)
			{
				cout << " merging in model " << c.second[i] << endl;
				modeler->mergeOtherVolume(*newObjModelers[c.second[i]].modeler, newObjModelers[c.second[i]].initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[newObjModelers[c.second[i]].initialFrame]].inverse() * newFramePosesWrtCurMap[frameTimes[modelerInfo.initialFrame]] * modelerInfo.initialCamPoseWrtMap.inverse());
#if 0 //visualize
				cv::Mat_<cv::Vec3b> mim = getPrettyIntelMapRender(*modeler, camParams, modelerInfo.initialCamPoseWrtMap * newFramePosesWrtCurMap[frameTimes[0]].inverse() * newFramePosesWrtCurMap[frameTimes.back()]);
				cv::imwrite((boost::format("mimnew%2%_%1%.png") % i % frameIndex).str(), mim);
#endif
				newObjModelers[c.second[i]].modeler.reset(); //save gpu ram
				modelIndicesToRemove.push_back(c.second[i]);
			}
			modelsToAdd.push_back(modelerInfo);
		}
	std::sort(modelIndicesToRemove.begin(), modelIndicesToRemove.end()); //required for removeFromRange()
	newObjModelers.erase(removeFromRange(newObjModelers, modelIndicesToRemove), newObjModelers.end());
	for(size_t i = 0; i < modelsToAdd.size(); i++) cout << "adding merged new model " << (newObjModelers.size() + i) << endl;
	newObjModelers.insert(newObjModelers.end(), modelsToAdd.begin(), modelsToAdd.end());

	/*
	 * push models that haven't been updated recently off the gpu (they'll be automatically pulled back in when needed)
	 */
	size_t numModelsOnGPU = 0;
	for(newObjModelerInfo& modelerInfo : newObjModelers)
		if(modelerInfo.lastUpdatedFrame + modelInactivityDeallocationPeriod < frameIndex)
			modelerInfo.modeler->deallocateBuffers();
		else
			numModelsOnGPU++;
	cout << numModelsOnGPU << " new obj models are in vram" << endl;
}

	t.stop("update and create new-obj modelers");
}

/*
 * auxiliary to nextFrame()
 */
void onlineObjModeler::processSegmentationDemonstration(rgbdFrame& frame)
{
	//TODO
#ifdef NOT_YET
	if(inDemoMode)
	{
		diff wrt the demo bkgnd map
		add to the demo bkgnd map
		if # high-p(m) pixels is very close (100?) to its avg over the last 50/* TODO ? */ frames
		{
			save demo end frame and its index
			get start & end obj segms
			use start segm to split/merge existing obj models
			fit rigid motion
			start an obj modeler for the obj if we didn't have one
			inDemoMode = false;
		}
	}
	else
	{
		if # high-p(m) pixels > 1000
		{
			save demo start frame and its index
			start a separate demo bkgnd map
			inDemoMode = true;
		}
	}
#endif
}

/*
 * call for each frame
 *
 * return false on error
 */
bool onlineObjModeler::nextFrameAux(rgbdFrame& frame, const boost::posix_time::ptime& frameTime)
{
	rgbd::timer t;
	const bool alignmentSuccess = onlineSceneDifferencer::alignAndDifference(frame, frameTime);
	t.stop("run alignAndDifference");
	if(!alignmentSuccess) return false;

	/*
	 * when a demo ends, update the relevant obj pose
	 *
	 * 20131206 TODO I might implement this in the active modeling class instead, to take advantage of extra info about expected camera pose
	 */
//	processSegmentationDemonstration(frame);

	if(doObjModeling)
	{
		t.restart();
		createAndUpdateObjModels(frame);
		t.stop("run createAndUpdate");
	}

	t.restart();
	onlineSceneDifferencer::updateMaps(frame, frameTime);
	t.stop("run updateMaps");

	t.restart();
	onlineSceneDifferencer::finishFrame(frameTime);
	t.stop("run finishFrame");
	return true;
}
