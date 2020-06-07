/*
 * locateMostMovingAreaInVideoZachFlow: use scene flow to constantly detect what part of the scene has the most motion (the goal is to do this quickly to direct view selection)
 *
 * Evan Herbst
 * 4 / 11 / 14
 */

#define USE_BAG //else take in two imgs only

#include <cassert>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <memory>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#ifdef USE_BAG
#include <cv_bridge/cv_bridge.h>
#endif
#include "rgbd_util/assert.h"
#include "rgbd_util/timer.h"
#include "rgbd_util/mathUtils.h"
#include "rgbd_util/CameraParams.h"
#include "rgbd_util/primesensorUtils.h"
//#include "rgbd_depthmaps/decompressDepth.h"
//#include "rgbd_depthmaps/depthIO.h"
//#include "rgbd_depthmaps/conversions.h"
#ifdef USE_BAG
#include "rgbd_bag_utils/contents.h"
#include "rgbd_bag_utils/rgbdBagReader2.h"
#endif
#include "image_features/halfDiskFeatures.h"
#include "optical_flow_utils/flow2color.h"
#include "rgbd_flow/rgbdFrameUtils.h"
#include "zachflow/zachFlowRunner.h"
using std::vector;
using std::string;
using std::cout;
using std::endl;
using std::ofstream;
using boost::lexical_cast;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
	po::options_description desc("Options");
	desc.add_options()
#ifdef USE_BAG
		("bag-path,b", po::value<fs::path>(), "for input")
		("bag-start-frame,a", po::value<uint32_t>()->default_value(0), "")
		("bag-frameskip,k", po::value<uint32_t>()->default_value(0), "")
		("bag-end-frame,n", po::value<uint32_t>()->default_value(1000000), "")
#else
		//TODO debugging 20140421
		("img0", po::value<fs::path>(), "")
		("img1", po::value<fs::path>(), "")
#endif

		/*
		 * 20140402 experiment on 10 runs of flow on different intervals of robotPushMultiobj1 containing movement: w=80 runs .08 - .32s; w=160 runs .2 - .6s; w=320 runs .8 - 2.5s
		 */
		("max-input-width,w", po::value<size_t>()->default_value(160), "frames given to flow are first downsized to this resolution")

		("flow-frame-interval,i", po::value<size_t>()->default_value(1), "how many frames ago we run flow wrt (>= 1)")
		("run-fwd", po::value<bool>()->default_value(true), "whether to run forward flow")
		("run-bkwd", po::value<bool>()->default_value(false), "whether to run backward flow")
		("outdir,o", po::value<fs::path>(), "")
		;
	po::variables_map vars;
	po::store(po::command_line_parser(argc, argv).options(desc).run(), vars);
	po::notify(vars);

	const size_t maxFlowInputFrameWidth = vars["max-input-width"].as<size_t>();
	const size_t flowFrameInterval = vars["flow-frame-interval"].as<size_t>();
	ASSERT_ALWAYS(flowFrameInterval >= 1);
	const bool runFwdFlow = vars["run-fwd"].as<bool>(), runBkwdFlow = vars["run-bkwd"].as<bool>();
	const fs::path outdir = vars["outdir"].as<fs::path>();
	fs::create_directories(outdir);

	const rgbd::CameraParams camParams = primesensor::getColorCamParams(rgbd::KINECT_640_DEFAULT);
	std::shared_ptr<openglContext> glContext(new openglContext(camParams.xRes, camParams.yRes)); //allocate the max size necessary for any rendering

#ifdef USE_BAG
	const fs::path bagFilepath = vars["bag-path"].as<fs::path>();
	const unsigned int startFrameIndex = vars["bag-start-frame"].as<uint32_t>(),
		frameskip = vars["bag-frameskip"].as<uint32_t>(), //use 0 to process every frame
		endFrameIndex = vars["bag-end-frame"].as<uint32_t>();

	//store N previous frames so we can always run flow wrt N frames ago
	std::unordered_map<size_t, cv::Mat_<cv::Vec3b>> imgs; //indexed by frame #
	std::unordered_map<size_t, cv::Mat_<float>> depthMaps; //indexed by frame #

	int32_t prevBestSubwindowX0 = -1, prevBestSubwindowY0 = -1;
#endif

	std::shared_ptr<zachFlowRunner> zachFlower, zachFlowerSub; //for coarse and subwindow flow, respectively (imgs have different sizes for the two cases so need separate flow calculators)
	const float lambda = 10;//20; //TODO ?
	const uint32_t numFlowIters = 100;//200; //TODO ?

#ifdef USE_BAG
	rgbd::rgbdBagReader2 frameReader(bagFilepath, startFrameIndex, endFrameIndex, frameskip, 0/* num prev frames to keep */);
	ASSERT_ALWAYS(frameReader.readOneFrame());
	uint32_t frameIndex = frameReader.getLastFrameIndex();
	sensor_msgs::ImageConstPtr imgMsg = frameReader.getLastImgPtr();
	cv_bridge::CvImageConstPtr curCIMsg = cv_bridge::toCvShare(imgMsg);
	cv::Mat_<cv::Vec3b>& curImg = imgs[frameIndex];
	cv::Mat_<float>& curDepth = depthMaps[frameIndex];
	curImg = curCIMsg->image.clone();
	curDepth = frameReader.getLastDepthImg();
	while(frameReader.readOneFrame())
	{
		rgbd::timer t;
		frameIndex = frameReader.getLastFrameIndex();
		cout << "on frame " << frameIndex << endl;

		cv::Mat_<cv::Vec3b>& curImg = imgs[frameIndex];
		cv::Mat_<float>& curDepth = depthMaps[frameIndex];
		imgMsg = frameReader.getLastImgPtr();
		cv_bridge::CvImageConstPtr curCIMsg = cv_bridge::toCvShare(imgMsg);
		curImg = curCIMsg->image.clone();
		curDepth = frameReader.getLastDepthImg();

		if(imgs.find(frameIndex - flowFrameInterval) != imgs.end()) //if we're far enough into the video
		{
			cv::Mat_<cv::Vec3b>& prevImg = imgs[frameIndex - flowFrameInterval];
			cv::Mat_<float>& prevDepth = depthMaps[frameIndex - flowFrameInterval];
#else
	cv::Mat_<cv::Vec3b> prevImg = cv::imread(vars["img0"].as<fs::path>().string()), curImg = cv::imread(vars["img1"].as<fs::path>().string());
	cv::Mat_<float> prevDepth(480, 640, 0.0f), curDepth(480, 640, 0.0f);
	const size_t frameIndex = 0;
#endif

			/*
			 * run flow at a coarse scale for subwindow detection
			 */
			rgbd::timer t;
			//downsize imgs so I can run optimization for more iters
			cv::Mat_<cv::Vec3b> prevImgSmall = prevImg.clone(), curImgSmall = curImg.clone();
			cv::Mat_<float> prevDepthSmall = prevDepth.clone(), curDepthSmall = curDepth.clone();
			while(prevImgSmall.rows > maxFlowInputFrameWidth) downsizeFrame(prevImgSmall, prevDepthSmall);
			while(curImgSmall.rows > maxFlowInputFrameWidth) downsizeFrame(curImgSmall, curDepthSmall);
			cv::Mat_<cv::Vec3f> coarseFlow(prevImgSmall.rows, prevImgSmall.cols);
			if(!zachFlower) zachFlower.reset(new zachFlowRunner(prevImgSmall.cols, prevImgSmall.rows, glContext));
			cv::Mat_<cv::Vec2f> zachflowResult = zachFlower->run(prevImgSmall, curImgSmall, lambda, numFlowIters);
			for(size_t i = 0; i < prevImgSmall.rows; i++)
				for(size_t j = 0; j < prevImgSmall.cols; j++)
				{
					const cv::Vec2f z = zachflowResult(i, j);
					coarseFlow(i, j) = cv::Vec3f(z[0], z[1], 0/* TODO change */);
				}
			t.stop("run coarse flow on whole img");

			cv::Mat_<cv::Vec3b> flowColsXYCoarse;
			cv::Mat_<cv::Vec3b> flowColsZCoarse;
			/*
			 * visualize
			 */
		{
			cv::Mat flowXY(coarseFlow.rows, coarseFlow.cols, cv::DataType<cv::Vec2f>::type), flowZ(coarseFlow.rows, coarseFlow.cols, cv::DataType<float>::type);
			vector<cv::Mat> src = {coarseFlow}, dest = {flowXY, flowZ};
			const int ch[] = {0, 0, 1, 1, 2, 2};
			cv::mixChannels(src, dest, ch, 3);
			double maxFlowXY;
			std::tie(flowColsXYCoarse, maxFlowXY) = flow2color(flowXY, 1/* max flow */);
			cv::Mat_<cv::Vec2f> flowZ2d(coarseFlow.size());
			for(uint32_t i = 0; i < coarseFlow.rows; i++)
				for(uint32_t j = 0; j < coarseFlow.cols; j++)
				{
					cv::Vec2f& f = flowZ2d.at<cv::Vec2f>(i, j);
					f[0] = flowZ.at<float>(i, j);
					f[1] = 0;
				}
			double maxFlowZ;
			std::tie(flowColsZCoarse, maxFlowZ) = flow2color(flowZ2d);
			cv::imwrite((outdir / (boost::format("coarseFlowXY%1%.png") % frameIndex).str()).string(), flowColsXYCoarse);
			cv::imwrite((outdir / (boost::format("coarseFlowZ%1%.png") % frameIndex).str()).string(), flowColsZCoarse);
		}

			/*
			 * find the best subwindow to run fine-scale flow on
			 *
			 * use count of large-flow pixels in window
			 */
			boost::multi_array<float, 2> flowMagnitude(boost::extents[prevImgSmall.rows][prevImgSmall.cols]); //at each pixel, in m
			const float sizeFactor = (float)camParams.xRes / prevImgSmall.cols; //downsampling factor, >= 1
			for(size_t i = 0; i < prevImgSmall.rows; i++)
				for(size_t j = 0; j < prevImgSmall.cols; j++)
				{
					const cv::Vec3f f = coarseFlow(i, j);
					const float estimatedDepth = (prevDepth(i, j) <= 0) ? 1 : prevDepth(i, j);
					flowMagnitude[i][j] = sqrt(sqr(f[0] * sizeFactor * estimatedDepth / camParams.focalLength) + sqr(f[1] * sizeFactor * estimatedDepth / camParams.focalLength) + sqr(f[2] * sizeFactor));
				}
			boost::multi_array<uint32_t, 2> flowLargeMagnitudeCountImg(boost::extents[prevImgSmall.rows][prevImgSmall.cols]); //at each pixel, whether the flow magnitude is above a threshold
			for(size_t i = 0; i < prevImgSmall.rows; i++)
				for(size_t j = 0; j < prevImgSmall.cols; j++)
				{
					flowLargeMagnitudeCountImg[i][j] = (flowMagnitude[i][j] > .003/* meters */) ? 1 : 0;
				}
			boost::multi_array<float, 2> flowMagnitudeIntegralImg;
			computeIntegralImage(flowMagnitude/*flowLargeMagnitudeCountImg*/, flowMagnitudeIntegralImg);
			const size_t subwindowWidth = 240,//prevImgSmall.cols,
				subwindowHeight = 180;//prevImgSmall.rows; //size in full-size img; TODO ?
			const size_t reducedSubwindowWidth = (size_t)rint(subwindowWidth / sizeFactor), reducedSubwindowHeight = (size_t)rint(subwindowHeight / sizeFactor);
			//evaluate each window of the proper size in the magnitude img
			float maxFlowMagnitudeSum = -1;
			size_t bestSubwindowX0 = -1, bestSubwindowY0 = -1; //in the coarse-flow img
			for(int32_t i = 0; i < prevImgSmall.rows - reducedSubwindowHeight; i++)
				for(int32_t j = 0; j < prevImgSmall.cols - reducedSubwindowWidth; j++)
				{
					const float subwindowSum = flowMagnitudeIntegralImg[i + reducedSubwindowHeight - 1][j + reducedSubwindowWidth - 1] - flowMagnitudeIntegralImg[i - 1][j + reducedSubwindowWidth - 1] - flowMagnitudeIntegralImg[i + reducedSubwindowHeight - 1][j - 1] + flowMagnitudeIntegralImg[i - 1][j - 1];
					if(subwindowSum > maxFlowMagnitudeSum)
					{
						maxFlowMagnitudeSum = subwindowSum;
						bestSubwindowX0 = j;
						bestSubwindowY0 = i;
					}
				}

#ifdef USE_BAG
			if(maxFlowMagnitudeSum < 10/* TODO ? */ && prevBestSubwindowX0 >= 0)
			{
				bestSubwindowX0 = prevBestSubwindowX0;
				bestSubwindowY0 = prevBestSubwindowY0;
			}
#endif

			/*
			 * run flow at a fine scale for motion estimation
			 */
			t.restart();
			const cv::Range colRange(rint(sizeFactor * bestSubwindowX0), rint(sizeFactor * (bestSubwindowX0 + reducedSubwindowWidth))), rowRange(rint(sizeFactor * bestSubwindowY0), rint(sizeFactor * (bestSubwindowY0 + reducedSubwindowHeight)));
			const cv::Mat_<cv::Vec3b> prevColorSubimg = prevImg(rowRange, colRange).clone(), curColorSubimg = curImg(rowRange, colRange).clone();
			const cv::Mat_<float> prevDepthSubimg = prevDepth(rowRange, colRange).clone(), curDepthSubimg = curDepth(rowRange, colRange).clone();
			cv::Mat_<cv::Vec3f> highResFlow(prevColorSubimg.size());
		{
			if(!zachFlowerSub) zachFlowerSub.reset(new zachFlowRunner(prevColorSubimg.cols, prevColorSubimg.rows, glContext));
			cv::Mat_<cv::Vec2f> zachflowResult = zachFlowerSub->run(prevColorSubimg, curColorSubimg, lambda, numFlowIters);
			for(size_t i = 0; i < zachflowResult.rows; i++)
				for(size_t j = 0; j < zachflowResult.cols; j++)
				{
					const cv::Vec2f z = zachflowResult(i, j);
					highResFlow(i, j) = cv::Vec3f(z[0], z[1], 0/* TODO change */);
				}
		}
			t.stop("run flow on subwindow");

			/*
			 * visualize
			 */
		{
			cv::Mat flowXY(highResFlow.rows, highResFlow.cols, cv::DataType<cv::Vec2f>::type), flowZ(highResFlow.rows, highResFlow.cols, cv::DataType<float>::type);
			vector<cv::Mat> src = {highResFlow}, dest = {flowXY, flowZ};
			const int ch[] = {0, 0, 1, 1, 2, 2};
			cv::mixChannels(src, dest, ch, 3);
			cv::Mat_<cv::Vec3b> flowColsXYSubimg;
			double maxFlowXY;
			std::tie(flowColsXYSubimg, maxFlowXY) = flow2color(flowXY, 1/* max flow */);
			cv::Mat_<cv::Vec2f> flowZ2d(highResFlow.size());
			for(uint32_t i = 0; i < highResFlow.rows; i++)
				for(uint32_t j = 0; j < highResFlow.cols; j++)
				{
					cv::Vec2f& f = flowZ2d.at<cv::Vec2f>(i, j);
					f[0] = flowZ.at<float>(i, j);
					f[1] = 0;
				}
			cv::Mat_<cv::Vec3b> flowColsZSubimg;
			double maxFlowZ;
			std::tie(flowColsZSubimg, maxFlowZ) = flow2color(flowZ2d);
			cv::Mat_<cv::Vec3b> flowColsXYImg(camParams.yRes, camParams.xRes), flowColsZImg(camParams.yRes, camParams.xRes);
			cv::resize(flowColsXYCoarse, flowColsXYImg, flowColsXYImg.size());
			cv::resize(flowColsZCoarse, flowColsZImg, flowColsZImg.size());
			flowColsXYSubimg.copyTo(flowColsXYImg(rowRange, colRange));
			flowColsZSubimg.copyTo(flowColsZImg(rowRange, colRange));

			cv::putText(flowColsXYImg, boost::lexical_cast<std::string>((int)rint(maxFlowMagnitudeSum)), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

			//pasted into the original img
			cv::imwrite((outdir / (boost::format("approxFlowXY%1%.png") % frameIndex).str()).string(), flowColsXYImg);
			cv::imwrite((outdir / (boost::format("approxFlowZ%1%.png") % frameIndex).str()).string(), flowColsZImg);
		}

#ifdef USE_BAG
			prevBestSubwindowX0 = bestSubwindowX0;
			prevBestSubwindowY0 = bestSubwindowY0;

			/*
			 * next steps:
			 * - run rigid mofitting and segmentation on at least the part of the scene containing the region observed to move (maybe can get away w/o the mrf)
			 * - mark unseen areas of the scene near the parts observed to move: these could be the back of the obj and we might want to direct the camera there
			 *  (but how do you define the scene you'll mark bits of?)
			 */

			//free memory
			imgs.erase(frameIndex - flowFrameInterval);
			depthMaps.erase(frameIndex - flowFrameInterval);
		}

		t.stop("process one frame");
	}
#endif

	return 0;
}
