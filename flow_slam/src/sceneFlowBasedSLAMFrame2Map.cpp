/*
 * sceneFlowBasedSLAMFrame2Map: frame-to-map use of flow rather than frame-to-frame
 *
 * 11 / 27 / 12
 */

#include <cassert>
#include <cstdint>
#include <string>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "rgbd_util/mathUtils.h"
#include "rgbd_util/kdtree2utils.h"
#include "xforms/xforms.h"
#include "rgbd_depthmaps/conversions.h"
#include "rgbd_depthmaps/decompressDepth.h"
#include "rgbd_depthmaps/depthIO.h"
#include "rgbd_depthmaps/depthMapImgProcessing.h"
#include "pcl_rgbd/cloudUtils.h"
#include "pcl_rgbd/cloudSearchTrees.h"
#include "pcl_rgbd/cloudTofroPLY.h"
#include "rgbd_bag_utils/contents.h"
#include "rgbd_bag_utils/rgbdBagReader2.h"
#include "point_cloud_icp/registration/icp_utility.h"
#include "optical_flow_utils/sceneFlowIO.h"
#include "rgbd_flow/runFlowTwoFrames.h"
#include "rgbd_flow/flowFieldEvaluation.h"
#include "rgbd_frame_common/rgbdFrame.h"
using std::string;
using std::cout;
using std::endl;
using boost::lexical_cast;
namespace fs = boost::filesystem;
using rgbd::eigen::Affine3f;

void visualizeFlowWithProjectedImgs(const cv::Mat& img1, const cv::Mat_<float>& depth1, const cv::Mat& img2, const cv::Mat_<float>& depth2, const cv::Mat& flow, const fs::path& outdirPlusFilebase)
{
	const fs::path outdir = outdirPlusFilebase.parent_path();
	fs::create_directories(outdir);
	cv::imwrite(outdirPlusFilebase.string() + "-img1.png", img1);
	cv::Mat_<cv::Vec3b> img2samples(img1.size());
	for(int32_t i = 0; i < img1.rows; i++)
		for(int32_t j = 0; j < img1.cols; j++)
		{
			const cv::Vec3f f = flow.at<cv::Vec3f>(i, j);
			cv::Mat_<cv::Vec3b> pix(1, 1);
			cv::getRectSubPix(img2, cv::Size(1, 1), cv::Point2f(clamp(j + f[0], 0.0f, (float)img1.cols - 1), clamp(i + f[1], 0.0f, (float)img1.rows - 1)), pix);
			img2samples.at<cv::Vec3b>(i, j) = pix(0, 0);
		}
	cv::imwrite(outdirPlusFilebase.string() + "-img2sampled.png", img2samples);
	cv::Mat_<cv::Vec3b> diffImg(img1.size());
	for(int32_t i = 0; i < img1.rows; i++)
		for(int32_t j = 0; j < img1.cols; j++)
			for(int k = 0; k < 3; k++)
				diffImg(i, j)[k] = rint(fabs((float)img1.at<cv::Vec3b>(i, j)[k] - (float)img2samples.at<cv::Vec3b>(i, j)[k]));
	cv::imwrite(outdirPlusFilebase.string() + "-img2diff.png", diffImg);
}

int main(int argc, char* argv[])
{
	uint32_t _ = 1;
	const fs::path inbagpath(argv[_++]);
	const uint32_t startFrame = lexical_cast<uint32_t>(argv[_++]), frameskip = lexical_cast<uint32_t>(argv[_++]), endFrame = lexical_cast<uint32_t>(argv[_++]);
	const double smoothnessWeight = lexical_cast<double>(argv[_++]);
	const float depthVsColorDataWeight = 1;
	const OpticalFlow::sceneFlowRegularizationType regularizationType = OpticalFlow::sceneFlowRegularizationType::HPB;
	const fs::path outdir(argv[_++]);
	fs::create_directories(outdir);

	const fs::path xformlistFilepath = outdir / "xformsToKeyframes.txt", xformlistTo0Filepath = outdir / "globalXforms.txt";

	const rgbd::CameraParams camParams = primesensor::getColorCamParams(rgbd::KINECT_640_DEFAULT);

	string depthTopic, imgTopic, cloudTopic;
	rgbd::estimateRGBDBagSchema(inbagpath, depthTopic, imgTopic, cloudTopic);
	rgbd::rgbdBagReader2 frameReader(inbagpath, depthTopic, imgTopic, cloudTopic, startFrame, endFrame, frameskip, 1/* num prev frames to keep */);

	ASSERT_ALWAYS(frameReader.readOneFrame());
	rgbdFrame prevFrame(camParams);
	prevFrame.name = lexical_cast<std::string>(frameReader.getLastFrameIndex());
	prevFrame.getImgMsgRef() = *cv_bridge::toCvCopy(frameReader.getLastImg(), "bgr8");
	prevFrame.getDepthImgRef() = frameReader.getLastDepthImg();

	rgbdFrame lastKeyframe = prevFrame;
	uint32_t lastKeyframeIndex = frameReader.getLastFrameIndex();
	ASSERT_ALWAYS(lastKeyframeIndex == startFrame);
	std::unordered_map<uint32_t, rgbd::eigen::Affine3f> keyframeXformsToFrame0;
	keyframeXformsToFrame0[lastKeyframeIndex] = rgbd::eigen::Affine3f::Identity();

	std::unordered_map<uint32_t, cv::Mat> flowFieldsToKeyframe; //src frame -> flow to prev keyframe; if src frame is a keyframe, we don't also store its flow field to itself

	std::vector<std::pair<ros::Time, rgbd::eigen::Affine3f> > xformsToKeyframe, xformsToFrame0;
	xformsToKeyframe.push_back(std::make_pair(prevFrame.getImgMsg().header.stamp, Affine3f::Identity()));
	xf::write_transforms_to_file(xformlistFilepath.string(), xformsToKeyframe);
	xformsToFrame0.push_back(std::make_pair(prevFrame.getImgMsg().header.stamp, Affine3f::Identity()));
	xf::write_transforms_to_file(xformlistTo0Filepath.string(), xformsToFrame0);

	while(frameReader.readOneFrame())
	{
		const uint32_t frameIndex = frameReader.getLastFrameIndex();

		rgbdFrame curFrame(camParams);
		curFrame.name = lexical_cast<std::string>(frameIndex);
		curFrame.getImgMsgRef() = *cv_bridge::toCvCopy(frameReader.getLastImg(), "bgr8");
		curFrame.getDepthImgRef() = frameReader.getLastDepthImg();

		const cv::Mat& prevImg = prevFrame.getImgMsg().image, &curImg = curFrame.getImgMsg().image;
		const cv::Mat_<float>& prevDepthImg = prevFrame.getDepthImg(), &curDepthImg = curFrame.getDepthImg();
		const pcl::PointCloud<rgbd::pt>& prevCloud = *prevFrame.getCloud(), &curCloud = *curFrame.getCloud();

		const cv::Mat& keyframeImg = lastKeyframe.getImgMsg().image;
		const cv::Mat_<float>& keyframeDepthImg = lastKeyframe.getDepthImg();

		/*
		 * get the flow from this frame to the first frame
		 */
		cv::Mat flowMatToKeyframe;
		const fs::path outpath = outdir / (boost::format("flowFrom%1%ToPrev.flo3") % frameIndex).str(),
			outpathToKeyframe = outdir / (boost::format("flowFrom%1%ToKeyframe.flo3") % frameIndex).str(),
			outpathToKeyframeReopt = outdir / (boost::format("flowFrom%1%ToKeyframeReopt.flo3") % frameIndex).str();
		if(fs::exists(outpathToKeyframe))
		{
			flowMatToKeyframe = readSceneFlow(outpathToKeyframe);
		}
		else
		{

#define RUN_SMALL_FLOW //run flow at half size for speed

			/*
			 * run flow vs prev frame
			 */

#ifdef RUN_SMALL_FLOW
			//downsample the frame
			cv::Size smallSize(prevImg.cols / 2, prevImg.rows / 2);
			cv::Mat_<cv::Vec3b> prevImgSmall, curImgSmall;
			cv::resize(prevImg, prevImgSmall, smallSize);
			cv::resize(curImg, curImgSmall, smallSize);
			cv::Mat_<float> prevDepthImgSmall, curDepthImgSmall;
			rgbd::downsampleDepthImg(prevDepthImg, prevDepthImgSmall, smallSize);
			rgbd::downsampleDepthImg(curDepthImg, curDepthImgSmall, smallSize);
#endif

			/*
			 * run flow
			 */

			cv::Mat f2fFlowMat; //cur -> prev frame
	#ifdef RUN_SMALL_FLOW
			for(int i = 0; i < curDepthImgSmall.rows; i++)
				for(int j = 0; j < curDepthImgSmall.cols; j++)
				{
					ASSERT_ALWAYS(!std::isnan(curDepthImgSmall(i, j)));
					ASSERT_ALWAYS(!std::isinf(curDepthImgSmall(i, j)));
					ASSERT_ALWAYS(fabs(curDepthImgSmall(i, j)) < 11);
					ASSERT_ALWAYS(!std::isnan(prevDepthImgSmall(i, j)));
					ASSERT_ALWAYS(!std::isinf(prevDepthImgSmall(i, j)));
					ASSERT_ALWAYS(fabs(prevDepthImgSmall(i, j)) < 11);
				}

			//TODO remove after debugging
			f2fFlowMat = runFlowTwoFramesSceneFlowExperiments(prevImgSmall, prevDepthImgSmall, curImgSmall, curDepthImgSmall, smoothnessWeight, depthVsColorDataWeight, regularizationType, boost::none/*estFlow*/, outdir / (boost::format("from%1%") % frameIndex).str());
			exit(0);

			f2fFlowMat = runFlowTwoFramesSceneFlowExperiments(curImgSmall, curDepthImgSmall, prevImgSmall, prevDepthImgSmall, smoothnessWeight, depthVsColorDataWeight, regularizationType, boost::none/*estFlow*/, outdir / (boost::format("from%1%") % frameIndex).str());
			//f2fFlowMat = runFlowTwoFramesCeliu2dto3d(curImgSmall, curDepthImgSmall, prevImgSmall, prevDepthImgSmall, smoothnessWeight, outdir, (boost::format("%1%") % frameIndex).str());
	#else
			f2fFlowMat = runFlowTwoFramesSceneFlowExperiments(prevImg, prevDepthImg, curImg, curDepthImg, smoothnessWeight, outdir, (boost::format("from%1%") % frameIndex).str());
	#endif
			ASSERT_ALWAYS(f2fFlowMat.type() == cv::DataType<cv::Vec3f>::type);
			for(int i = 0; i < f2fFlowMat.rows; i++)
				for(int j = 0; j < f2fFlowMat.cols; j++)
				{
					cv::Vec3f f = f2fFlowMat.at<cv::Vec3f>(i, j);
					for(int k = 0; k < 3; k++)
					{
						ASSERT_ALWAYS(!std::isnan(f[k]));
						ASSERT_ALWAYS(!std::isinf(f[k]));
					}
				}

#ifdef RUN_SMALL_FLOW
		{
			//upsample the flow
			cv::Mat_<cv::Vec3f> flowMatLarge;
			cv::resize(f2fFlowMat, flowMatLarge, prevImg.size());
			const float factor = (float)flowMatLarge.rows / f2fFlowMat.rows;
			for(uint32_t i = 0; i < (uint32_t)flowMatLarge.rows; i++)
				for(uint32_t j = 0; j < (uint32_t)flowMatLarge.cols; j++)
				{
					cv::Vec3f& f = flowMatLarge(i, j);
					f[0] *= factor;
					f[1] *= factor;
				}
			f2fFlowMat = flowMatLarge;
		}
#endif

			/*
			 * visualize flow
			 */
			writeSceneFlow(f2fFlowMat, outpath);

			visualizeFlowWithProjectedImgs(curImg, curDepthImg, prevImg, prevDepthImg, f2fFlowMat, outdir / "flowvis" / (boost::format("frame%1%") % frameIndex).str());

			/*
			 * get an initial estimate of the flow to the first frame by adding consecutive-frame flow to prev-frame-to-first-frame flow
			 */
			if(lastKeyframe.name == prevFrame.name) //if the flow from the prev frame to the prev keyframe is zero (ie they're the same frame), there won't be an entry in flowFieldsToKeyframe
			{
				flowMatToKeyframe = f2fFlowMat.clone();
			}
			else
			{
				flowMatToKeyframe.create(prevImg.rows, prevImg.cols, cv::DataType<cv::Vec3f>::type);
				const cv::Mat& flowPrevToKeyframe = flowFieldsToKeyframe[frameIndex - frameskip - 1];
				for(uint32_t i = 0; i < (uint32_t)prevImg.rows; i++)
					for(uint32_t j = 0; j < (uint32_t)prevImg.cols; j++)
					{
						//TODO better way to do this w/ inconsistent scene flow fields?

						//follow the flow to the prev frame
						Eigen::Vector2f p(j, i);
						const cv::Vec3f u = f2fFlowMat.at<cv::Vec3f>(i, j);
						p.x() += u[0];
						p.y() += u[1];
						//follow the flow to frame 0
						const int64_t ix = std::min((int64_t)prevImg.cols - 1, std::max((int64_t)0, (int64_t)p.x())), iy = std::min((int64_t)prevImg.rows - 1, std::max((int64_t)0, (int64_t)p.y())),
							ix2 = std::min(ix + 1, (int64_t)prevImg.cols - 1), iy2 = std::min(iy + 1, (int64_t)prevImg.rows - 1);
						const float ax = p.x() - ix, ay = p.y() - iy;
						const cv::Vec3f u00 = flowPrevToKeyframe.at<cv::Vec3f>(iy, ix), u01 = flowPrevToKeyframe.at<cv::Vec3f>(iy2, ix), u10 = flowPrevToKeyframe.at<cv::Vec3f>(iy, ix2), u11 = flowPrevToKeyframe.at<cv::Vec3f>(iy2, ix2);
						const float uk0 = linterp(linterp(u00[0], u01[0], ay), linterp(u10[0], u11[0], ay), ax),
							vk0 = linterp(linterp(u00[1], u01[1], ay), linterp(u10[1], u11[1], ay), ax),
							wk0 = linterp(linterp(u00[2], u01[2], ay), linterp(u10[2], u11[2], ay), ax);
						ASSERT_ALWAYS(!std::isnan(u[0])); ASSERT_ALWAYS(!std::isinf(u[0]));
						ASSERT_ALWAYS(!std::isnan(u[1])); ASSERT_ALWAYS(!std::isinf(u[1]));
						ASSERT_ALWAYS(!std::isnan(u[2])); ASSERT_ALWAYS(!std::isinf(u[2]));
						ASSERT_ALWAYS(!std::isnan(uk0)); ASSERT_ALWAYS(!std::isinf(uk0));
						ASSERT_ALWAYS(!std::isnan(vk0)); ASSERT_ALWAYS(!std::isinf(vk0));
						ASSERT_ALWAYS(!std::isnan(wk0)); ASSERT_ALWAYS(!std::isinf(wk0));
						flowMatToKeyframe.at<cv::Vec3f>(i, j) = cv::Vec3f(u[0] + uk0, u[1] + vk0, u[2] + wk0);
					}
			}

			writeSceneFlow(flowMatToKeyframe, outpathToKeyframe);
		}
		flowFieldsToKeyframe[frameIndex] = flowMatToKeyframe;
		flow::writeFlowFieldEvaluationImgs(camParams, curImg, curDepthImg, keyframeImg, keyframeDepthImg, flowFieldsToKeyframe[frameIndex], xfmt((outdir / (boost::format("%1%beforeReopt-%%x%%.png") % frameIndex).str()).string()));
		visualizeFlowWithProjectedImgs(curImg, curDepthImg, keyframeImg, keyframeDepthImg, flowFieldsToKeyframe[frameIndex], outdir / "flowvis" / (boost::format("frame%1%ToKey") % frameIndex).str());

#if 1 //debugging
		const auto CHECK = [](const cv::Mat& f)
			{
				ASSERT_ALWAYS(f.type() == cv::DataType<cv::Vec3f>::type);
				for(int i = 0; i < f.rows; i++)
					for(int j = 0; j < f.cols; j++)
						for(int k = 0; k < 3; k++)
						{
							ASSERT_ALWAYS(!std::isnan(f.at<cv::Vec3f>(i, j)[k]));
							ASSERT_ALWAYS(!std::isinf(f.at<cv::Vec3f>(i, j)[k]));
						}
			};
		CHECK(flowFieldsToKeyframe[frameIndex]);
#endif

		cv::Mat_<cv::Vec3f> reoptimizedFlowMat;
		if(fs::exists(outpathToKeyframeReopt))
		{
			reoptimizedFlowMat = readSceneFlow(outpathToKeyframeReopt);
		}
		else
		{
			/*
			 * reoptimize flow to keyframe
			 */

#ifdef RUN_SMALL_FLOW
			//downsample the frame
			cv::Size smallSize(prevImg.cols / 2, prevImg.rows / 2);
			cv::Mat_<cv::Vec3b> keyframeImgSmall, curImgSmall;
			cv::resize(keyframeImg, keyframeImgSmall, smallSize);
			cv::resize(curImg, curImgSmall, smallSize);
			cv::Mat_<float> keyframeDepthImgSmall, curDepthImgSmall;
			rgbd::downsampleDepthImg(keyframeDepthImg, keyframeDepthImgSmall, smallSize);
			rgbd::downsampleDepthImg(curDepthImg, curDepthImgSmall, smallSize);
			cv::Mat_<cv::Vec3f> flowFieldToKeyframeSmall;
			cv::resize(flowFieldsToKeyframe[frameIndex], flowFieldToKeyframeSmall, smallSize);
			const float factor = (float)flowFieldToKeyframeSmall.rows / flowFieldsToKeyframe[frameIndex].rows;
			for(uint32_t i = 0; i < (uint32_t)flowFieldToKeyframeSmall.rows; i++)
				for(uint32_t j = 0; j < (uint32_t)flowFieldToKeyframeSmall.cols; j++)
				{
					cv::Vec3f& f = flowFieldToKeyframeSmall(i, j);
					f[0] *= factor;
					f[1] *= factor;
				}
#endif

			reoptimizedFlowMat = runFlowTwoFramesSceneFlowExperiments(curImgSmall, curDepthImgSmall, keyframeImgSmall, keyframeDepthImgSmall, smoothnessWeight, depthVsColorDataWeight, regularizationType, flowFieldToKeyframeSmall, outdir / (boost::format("from%1%toKeyReopt") % frameIndex).str());

#ifdef RUN_SMALL_FLOW
		{
			//upsample the flow
			cv::Mat_<cv::Vec3f> flowMatLarge;
			cv::resize(reoptimizedFlowMat, flowMatLarge, prevImg.size());
			const float factor = (float)flowMatLarge.rows / reoptimizedFlowMat.rows;
			for(uint32_t i = 0; i < (uint32_t)flowMatLarge.rows; i++)
				for(uint32_t j = 0; j < (uint32_t)flowMatLarge.cols; j++)
				{
					cv::Vec3f& f = flowMatLarge(i, j);
					f[0] *= factor;
					f[1] *= factor;
				}
			reoptimizedFlowMat = flowMatLarge;
		}
#endif

			writeSceneFlow(reoptimizedFlowMat, outpathToKeyframeReopt);
		}
		flowFieldsToKeyframe[frameIndex] = reoptimizedFlowMat;
		flow::writeFlowFieldEvaluationImgs(camParams, curImg, curDepthImg, keyframeImg, keyframeDepthImg, flowFieldsToKeyframe[frameIndex], xfmt((outdir / (boost::format("%1%afterReopt-%%x%%.png") % frameIndex).str()).string()));
		visualizeFlowWithProjectedImgs(curImg, curDepthImg, keyframeImg, keyframeDepthImg, flowFieldsToKeyframe[frameIndex], outdir / "flowvis" / (boost::format("frame%1%ToKeyReopt") % frameIndex).str());

#if 1
		/*
		 * numerically compare un- and reoptimized flows
		 */
		cv::Mat_<float> bcErrorBefore = flow::evaluateBrightnessConstancy(curImg, keyframeImg, flowMatToKeyframe);
		cv::Mat_<float> bcErrorAfter = flow::evaluateBrightnessConstancy(curImg, keyframeImg, reoptimizedFlowMat);
		cv::Mat_<float> dcErrorBefore = flow::evaluateDepthConsistency(camParams, curDepthImg, keyframeDepthImg, flowMatToKeyframe);
		cv::Mat_<float> dcErrorAfter = flow::evaluateDepthConsistency(camParams, curDepthImg, keyframeDepthImg, reoptimizedFlowMat);
		cv::Mat_<float> lrErrorBefore = flow::evaluateLocalRigidity(camParams, curDepthImg, flowMatToKeyframe);
		cv::Mat_<float> lrErrorAfter = flow::evaluateLocalRigidity(camParams, curDepthImg, reoptimizedFlowMat);
		double bcImprovement = 0, dcImprovement = 0, lrImprovement = 0;
		for(uint32_t i = 0; i < (uint32_t)curImg.rows; i++)
			for(uint32_t j = 0; j < (uint32_t)curImg.cols; j++)
			{
				bcImprovement += bcErrorBefore(i, j) - bcErrorAfter(i, j);
				dcImprovement += dcErrorBefore(i, j) - dcErrorAfter(i, j);
				lrImprovement += lrErrorBefore(i, j) - lrErrorAfter(i, j);
			}
		cout << "improvements: " << bcImprovement << ' ' << dcImprovement << ' ' << lrImprovement << endl;
#endif

		/*
		 * compute a rigid motion for the whole scene to the last keyframe
		 */

		const pcl::PointCloud<rgbd::pt>& keyframeCloud = *lastKeyframe.getCloud();
		const boost::shared_ptr<kdtree2> keyframeTree = rgbd::createKDTree2(keyframeCloud);
		const kdtreeNbrhoodSpec nspec = kdtreeNbrhoodSpec::byCount(1);

		uint64_t numNanFlows = 0, numInfFlows = 0;
		std::deque<std::pair<rgbd::eigen::Vector3f, rgbd::eigen::Vector3f>> corrPts3d;
		for(uint32_t k = 0; k < curCloud.points.size(); k++)
		{
			const uint32_t i = curCloud.points[k].imgY, j = curCloud.points[k].imgX;
			const cv::Vec3f flow = flowMatToKeyframe.at<cv::Vec3f>(i, j);
			if(std::isnan(flow[0]) || std::isnan(flow[1]) || std::isnan(flow[2])) numNanFlows++;
			else if(std::isinf(flow[0]) || std::isinf(flow[1]) || std::isinf(flow[2])) numInfFlows++;
			else
			{
				const Eigen::Vector3f srcPt = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(curCloud.points[k]);
				//create a new 3-d point at the target time by adding src-time flow to src-time measured point
				const Eigen::Vector3f tgtPt = srcPt + rgbd::eigen::Vector3f(flow[0] * srcPt.z() / camParams.focalLength, flow[1] * srcPt.z() / camParams.focalLength, flow[2]);

				/*
				 * only include pts that flow to somewhere near an existing pt in the tgt frame (this check *does* improve xforms 20121209)
				 */
				const std::vector<float> qpt = {tgtPt[0], tgtPt[1], tgtPt[2]};
				const std::vector<kdtree2_result> nbrs = searchKDTree(*keyframeTree, nspec, qpt);
				ASSERT_ALWAYS(nbrs.size() == 1);
				if(nbrs[0].dis < sqr(.005/* TODO ? */ * primesensor::stereoErrorRatio(prevCloud.points[nbrs[0].idx].z))) //if tgtPt is near a prev-frame pt
					corrPts3d.push_back(std::make_pair(srcPt, tgtPt));
			}
		}
		cout << numNanFlows << " nan flows; " << numInfFlows << " inf flows; " << corrPts3d.size() << " corrs for xform fitting" << endl;

		std::vector<float> corrWeights(corrPts3d.size(), 1);
#define USE_DISTANCE_DOWNWEIGHTING
#ifdef USE_DISTANCE_DOWNWEIGHTING
		for(uint32_t k = 0; k < corrPts3d.size(); k++) corrWeights[k] = 1 / sqr(corrPts3d[k].first.z());
#endif
		rgbd::eigen::Quaternionf q;
		rgbd::eigen::Vector3f xlate;
		registration::runClosedFormAlignment(corrPts3d, corrWeights, q, xlate);

#if 0
		/*
		 * find inlier points to the fit xform and refit (TODO does this matter?)
		 */
		std::vector<rgbd::eigen::Vector4f, rgbd::eigen::aligned_allocator<rgbd::eigen::Vector4f> > srcPts(corrPts3d.size()), tgtPts(corrPts3d.size());
		std::vector<std::pair<int, int>> corrs(corrPts3d.size()), inlierCorrs;
		for(uint32_t i = 0; i < corrPts3d.size(); i++)
		{
			srcPts[i] = Eigen::Vector4f(corrPts3d[i].first[0], corrPts3d[i].first[1], corrPts3d[i].first[2], 1);
			tgtPts[i] = Eigen::Vector4f(corrPts3d[i].second[0], corrPts3d[i].second[1], corrPts3d[i].second[2], 1);
			corrs[i] = std::make_pair(i, i);
		}
		registration::getInlierCorrespondencesUniformWithDepth(srcPts, tgtPts, corrs, .01/* inlier dist at 1m; TODO ? */, rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(xlate) * q), inlierCorrs);
		cout << "refitting: got " << inlierCorrs.size() << " inliers of " << corrs.size() << endl;
		std::vector<uint32_t> inlierIndices(inlierCorrs.size());
		for(uint32_t i = 0; i < inlierCorrs.size(); i++) inlierIndices[i] = inlierCorrs[i].first;
		std::deque<std::pair<rgbd::eigen::Vector3f, rgbd::eigen::Vector3f>> inlierPts3d(inlierIndices.size());
		copySelected(corrPts3d, inlierIndices, inlierPts3d.begin());

		std::vector<float> inlierWeights(inlierIndices.size());
		copySelected(corrWeights, inlierIndices, inlierWeights.begin());
		registration::runClosedFormAlignment(inlierPts3d, inlierWeights, q, xlate);
#endif

		ASSERT_ALWAYS(keyframeXformsToFrame0.find(lastKeyframeIndex) != keyframeXformsToFrame0.end());
		const rgbd::eigen::Affine3f xformToKeyframe = rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(xlate) * q),
			xformToFrame0 = keyframeXformsToFrame0[lastKeyframeIndex] * xformToKeyframe;
		cout << "last keyframe (" << lastKeyframeIndex << ") xform to 0 is " << endl << keyframeXformsToFrame0[lastKeyframeIndex].matrix() << endl;
		cout << "xform " << frameIndex << " -> " << lastKeyframeIndex << ": " << endl << xformToKeyframe.matrix() << endl;
		cout << "xform to 0: " << endl << xformToFrame0.matrix() << endl;

		xformsToKeyframe.push_back(std::make_pair(curFrame.getImgMsg().header.stamp, xformToKeyframe));
		xf::write_transforms_to_file(xformlistFilepath.string(), xformsToKeyframe);
		xformsToFrame0.push_back(std::make_pair(curFrame.getImgMsg().header.stamp, xformToFrame0));
		xf::write_transforms_to_file(xformlistTo0Filepath.string(), xformsToFrame0);

#if 1 //debugging
	{
		pcl::PointCloud<rgbd::pt> flowedCloud = curCloud;
		for(uint32_t l = 0; l < flowedCloud.points.size(); l++)
		{
			rgbd::eigen::Vector3f x = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(flowedCloud.points[l]);
			const cv::Vec3f f = flowMatToKeyframe.at<cv::Vec3f>(flowedCloud.points[l].imgY, flowedCloud.points[l].imgX);
			x[0] += f[0] * flowedCloud.points[l].z / camParams.focalLength;
			x[1] += f[1] * flowedCloud.points[l].z / camParams.focalLength;
			x[2] += f[2];
			rgbd::eigen2ptX(flowedCloud.points[l], x);
		}
		rgbd::write_ply_file(flowedCloud, outdir / (boost::format("flowedFrom%1%ToKey.ply") % frameIndex).str());
	}
#endif

//		exit(0); //TODO remove

		prevFrame = curFrame;
		float dx, da; //in m and rad
		xf::transform_difference(Eigen::Affine3f::Identity(), xformToKeyframe, dx, da);
		cout << "frame " << frameIndex << ": dx = " << dx << ", da = " << da << endl;
		const bool makeNewKeyframe = false;//(dx > .5 || da > .3); //TODO ?
		if(makeNewKeyframe)
		{
			lastKeyframe = curFrame;
			lastKeyframeIndex = frameIndex;
			keyframeXformsToFrame0[frameIndex] = xformToFrame0;
		}
	}

	return 0;
}
