/*
 * sceneFlowBasedSLAM: do mapping of a single rigid scene using flow
 *
 * Evan Herbst
 * 11 / 5 / 12
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
#include "xforms/xforms.h"
#include "rgbd_depthmaps/conversions.h"
#include "pcl_rgbd/cloudUtils.h"
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

	const fs::path xformlistFilepath = outdir / "globalXforms.txt";

	const rgbd::CameraParams camParams = primesensor::getColorCamParams(rgbd::KINECT_640_DEFAULT);

	string depthTopic, imgTopic, cloudTopic;
	rgbd::estimateRGBDBagSchema(inbagpath, depthTopic, imgTopic, cloudTopic);
	rgbd::rgbdBagReader2 frameReader(inbagpath, depthTopic, imgTopic, cloudTopic, startFrame, endFrame, frameskip, 1/* num prev frames to keep */);

	ASSERT_ALWAYS(frameReader.readOneFrame());
	rgbdFrame prevFrame(camParams);
	prevFrame.getImgMsgRef() = *cv_bridge::toCvCopy(frameReader.getLastImg(), "bgr8");
	prevFrame.getDepthImgRef() = frameReader.getLastDepthImg();

	std::vector<std::pair<ros::Time, rgbd::eigen::Affine3f> > xforms;
	Affine3f prevXform = Affine3f::Identity();
	xforms.push_back(std::make_pair(prevFrame.getImgMsg().header.stamp, prevXform));
	xf::write_transforms_to_file(xformlistFilepath.string(), xforms);

	while(frameReader.readOneFrame())
	{
		const uint32_t frameIndex = frameReader.getLastFrameIndex();

		rgbdFrame curFrame(camParams);
		curFrame.getImgMsgRef() = *cv_bridge::toCvCopy(frameReader.getLastImg(), "bgr8");
		curFrame.getDepthImgRef() = frameReader.getLastDepthImg();

		const cv::Mat& prevImg = prevFrame.getImgMsg().image, &curImg = curFrame.getImgMsg().image;
		const cv::Mat_<float>& prevDepthImg = prevFrame.getDepthImg(), &curDepthImg = curFrame.getDepthImg();
		const pcl::PointCloud<rgbd::pt>& prevCloud = *prevFrame.getCloud(), &curCloud = *curFrame.getCloud();

		/*
		 * run flow
		 */

		cv::Mat flowMat;
#define RUN_SMALL_FLOW //run flow at half size for speed
#ifdef RUN_SMALL_FLOW
		const fs::path outpath = outdir / (boost::format("flow%1%.flo3") % frameIndex).str();
#else
		const fs::path outpath = outdir / (boost::format("flowLarge%1%.flo3") % frameIndex).str();
#endif
		if(fs::exists(outpath))
		{
			flowMat = readSceneFlow(outpath);
			cout << "read flow from '" << outpath << "'" << endl;
		}
		else
		{
			/*
			 * run flow
			 */

#ifdef RUN_SMALL_FLOW
			//downsample the frame
			cv::Size smallSize(prevImg.cols / 2, prevImg.rows / 2);
			cv::Mat_<cv::Vec3b> prevImgSmall, curImgSmall;
			cv::resize(prevImg, prevImgSmall, smallSize);
			cv::resize(curImg, curImgSmall, smallSize);
			cv::Mat_<float> prevDepthImgSmall, curDepthImgSmall;
			cv::resize(prevDepthImg, prevDepthImgSmall, smallSize);
			cv::resize(curDepthImg, curDepthImgSmall, smallSize);
#endif

#ifdef RUN_SMALL_FLOW
			flowMat = runFlowTwoFramesSceneFlowExperiments(prevImgSmall, prevDepthImgSmall, curImgSmall, curDepthImgSmall, smoothnessWeight, depthVsColorDataWeight, regularizationType, boost::none, outdir / (boost::format("%1%") % frameIndex).str());
			//flowMat = runFlowTwoFramesCeliu2dto3d(prevImgSmall, prevDepthImgSmall, curImgSmall, curDepthImgSmall, smoothnessWeight, outdir, (boost::format("%1%") % frameIndex).str());
#else
			flowMat = runFlowTwoFramesSceneFlowExperiments(prevImg, prevDepthImg, curImg, curDepthImg, smoothnessWeight, outdir, (boost::format("%1%") % frameIndex).str());
#endif
			ASSERT_ALWAYS(flowMat.type() == cv::DataType<cv::Vec3f>::type);

#ifdef RUN_SMALL_FLOW
			//upsample the flow
			cv::Mat_<cv::Vec3f> flowMatLarge;
			cv::resize(flowMat, flowMatLarge, prevImg.size());
			const float factor = (float)flowMatLarge.rows / flowMat.rows;
			for(uint32_t i = 0; i < (uint32_t)flowMatLarge.rows; i++)
				for(uint32_t j = 0; j < (uint32_t)flowMatLarge.cols; j++)
				{
					cv::Vec3f& f = flowMatLarge(i, j);
					f[0] *= factor;
					f[1] *= factor;
				}
			flowMat = flowMatLarge;
#endif

			/*
			 * visualize flow
			 */
			writeSceneFlow(flowMat, outpath);
		}

#if 1 //evaluate flow error
		const cv::Mat_<float> bceImg = flow::evaluateBrightnessConstancy(prevImg, curImg, flowMat);
		const cv::Mat_<float> dceImg = flow::evaluateDepthConsistency(camParams, prevDepthImg, curDepthImg, flowMat);
		const cv::Mat_<float> lrImg = flow::evaluateLocalRigidity(camParams, prevDepthImg, flowMat);
		cv::Mat_<uint8_t> bceImgUC(bceImg.rows, bceImg.cols);
		cv::Mat_<uint8_t> dceImgUC(dceImg.rows, dceImg.cols);
		cv::Mat_<uint8_t> lrImgUC(lrImg.rows, lrImg.cols);
		float bceMax = -FLT_MAX, dceMax = -FLT_MAX, lrMax = -FLT_MAX;
		for(uint32_t i = 0; i < (uint32_t)prevImg.rows; i++)
			for(uint32_t j = 0; j < (uint32_t)prevImg.cols; j++)
			{
				bceMax = std::max(bceMax, bceImg(i, j));
				dceMax = std::max(dceMax, dceImg(i, j));
				lrMax = std::max(lrMax, lrImg(i, j));
			}
		cout << "maxes " << bceMax << " " << dceMax << ' ' << lrMax << endl;
		if(dceMax > .01) dceMax = .01;
		for(uint32_t i = 0; i < (uint32_t)prevImg.rows; i++)
			for(uint32_t j = 0; j < (uint32_t)prevImg.cols; j++)
			{
				bceImgUC(i, j) = 255 * bceImg(i, j) / bceMax;
				dceImgUC(i, j) = 255 * dceImg(i, j) / dceMax;
				lrImgUC(i, j) = 255 * lrImg(i, j) / lrMax;
			}
		cv::imwrite((outdir / (boost::format("%1%-bce.png") % frameIndex).str()).string(), bceImgUC);
		cv::imwrite((outdir / (boost::format("%1%-dce.png") % frameIndex).str()).string(), dceImgUC);
		cv::imwrite((outdir / (boost::format("%1%-lr.png") % frameIndex).str()).string(), lrImgUC);
//		{int q; std::cin >> q;}
#endif

		/*
		 * compute a rigid motion for the whole scene
		 */

		std::deque<std::pair<rgbd::eigen::Vector3f, rgbd::eigen::Vector3f>> corrPts3d;
		for(uint32_t k = 0; k < prevCloud.points.size(); k++)
		{
			const uint32_t i = prevCloud.points[k].imgY, j = prevCloud.points[k].imgX;
			const cv::Vec3f flow = flowMat.at<cv::Vec3f>(i, j);
			if(!std::isnan(flow[2]) && !std::isinf(flow[2]))
			{
				const Eigen::Vector3f srcPt = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(prevCloud.points[k]);
				//create a new 3-d point at the target time by adding src-time flow to src-time measured point
				const Eigen::Vector3f tgtPt = srcPt + rgbd::eigen::Vector3f(flow[0] * srcPt.z() / camParams.focalLength, flow[1] * srcPt.z() / camParams.focalLength, flow[2]);
				corrPts3d.push_back(std::make_pair(srcPt, tgtPt));
			}
		}
		std::vector<float> corrWeights(corrPts3d.size(), 1);
#define USE_DISTANCE_DOWNWEIGHTING
#ifdef USE_DISTANCE_DOWNWEIGHTING
		for(uint32_t k = 0; k < corrPts3d.size(); k++) corrWeights[k] = 1 / sqr(corrPts3d[k].first.z());
#endif
		rgbd::eigen::Quaternionf q;
		rgbd::eigen::Vector3f xlate;
		registration::runClosedFormAlignment(corrPts3d, corrWeights, q, xlate);

		prevXform = prevXform * rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(xlate) * q).inverse();
		xforms.push_back(std::make_pair(curFrame.getImgMsg().header.stamp, prevXform));
		xf::write_transforms_to_file(xformlistFilepath.string(), xforms);

#if 1 //debugging
	{
		pcl::PointCloud<rgbd::pt> flowedCloud = prevCloud;
		for(uint32_t l = 0; l < flowedCloud.points.size(); l++)
		{
			rgbd::eigen::Vector3f x = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(flowedCloud.points[l]);
			const cv::Vec3f f = flowMat.at<cv::Vec3f>(flowedCloud.points[l].imgY, flowedCloud.points[l].imgX);
			x[0] += f[0] * flowedCloud.points[l].z / camParams.focalLength;
			x[1] += f[1] * flowedCloud.points[l].z / camParams.focalLength;
			x[2] += f[2];
			rgbd::eigen2ptX(flowedCloud.points[l], x);
		}
		rgbd::write_ply_file(flowedCloud, outdir / (boost::format("flowedTo%1%.ply") % frameIndex).str());

		pcl::PointCloud<rgbd::pt> xformedCloud = prevCloud;
		rgbd::transform_point_cloud_in_place(rgbd::eigen::Affine3f(rgbd::eigen::Translation3f(xlate) * q), xformedCloud, true/* normals */);
		rgbd::write_ply_file(xformedCloud, outdir / (boost::format("xformedTo%1%.ply") % frameIndex).str());

		rgbd::write_ply_file(curCloud, outdir / (boost::format("curCloud%1%.ply") % frameIndex).str());
	}
#endif

		prevFrame = curFrame;
	}

	//then use the partial map viewer

	return 0;
}
