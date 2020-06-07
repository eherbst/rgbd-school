/*
 * dficpBasedVO: visual odometry with damn fast icp
 *
 * Evan Herbst
 * 11 / 14 / 12
 */

#include <cassert>
#include <cstdint>
#include <string>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "rgbd_util/timer.h"
#include "rgbd_util/mathUtils.h"
#include "xforms/xforms.h"
#include "rgbd_depthmaps/conversions.h"
#include "pcl_rgbd/cloudUtils.h"
#include "pcl_rgbd/cloudTofroPLY.h"
#include "rgbd_bag_utils/contents.h"
#include "rgbd_bag_utils/rgbdBagReader2.h"
#include "range_image_icp/motion_estimation_ri.h"
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

	registration::RangeImageMotionEstimation icp(8/* # threads */);
	boost::shared_ptr<pcl::RangeImagePlanar> pclRangeImg(new pcl::RangeImagePlanar);
	std::vector<float> depthBuf(prevFrame.getDepthImg().rows * prevFrame.getDepthImg().cols);
	for(uint32_t i = 0, l = 0; i < (uint32_t)prevFrame.getDepthImg().rows; i++)
		for(uint32_t j = 0; j < (uint32_t)prevFrame.getDepthImg().cols; j++, l++)
			depthBuf[l] = prevFrame.getDepthImg()(i, j);
	pclRangeImg->setDepthImage(depthBuf.data(), camParams.xRes, camParams.yRes, camParams.centerX, camParams.centerY, camParams.focalLength, camParams.focalLength);
	icp.addObservation(pclRangeImg);

	while(frameReader.readOneFrame())
	{
		rgbdFrame curFrame(camParams);
		curFrame.getImgMsgRef() = *cv_bridge::toCvCopy(frameReader.getLastImg(), "bgr8");
		curFrame.getDepthImgRef() = frameReader.getLastDepthImg();

		const cv::Mat_<float>& curDepth = curFrame.getDepthImg();

		/*
		 * run icp
		 */

		rgbd::timer t;
		for(uint32_t i = 0, l = 0; i < (uint32_t)curDepth.rows; i++)
			for(uint32_t j = 0; j < (uint32_t)curDepth.cols; j++, l++)
				depthBuf[l] = curDepth(i, j);
		pclRangeImg->setDepthImage(depthBuf.data(), camParams.xRes, camParams.yRes, camParams.centerX, camParams.centerY, camParams.focalLength, camParams.focalLength);
		icp.addObservation(pclRangeImg);
		const Eigen::Affine3f xform = icp.getCurrentTransformation();
		t.stop("run icp");

		prevXform = xform;
		xforms.push_back(std::make_pair(curFrame.getImgMsg().header.stamp, prevXform));
		xf::write_transforms_to_file(xformlistFilepath.string(), xforms);

		prevFrame = curFrame;
	}

	//then use the partial map viewer

	return 0;
}
