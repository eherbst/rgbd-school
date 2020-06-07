/*
 * icpBasedVO: use our icp code to do visual odometry
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
#include "point_cloud_icp/registration/icp_combined.h"
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

	while(frameReader.readOneFrame())
	{
		rgbdFrame curFrame(camParams);
		curFrame.getImgMsgRef() = *cv_bridge::toCvCopy(frameReader.getLastImg(), "bgr8");
		curFrame.getDepthImgRef() = frameReader.getLastDepthImg();

		const pcl::PointCloud<rgbd::pt>& prevCloud = *prevFrame.getCloud(), &curCloud = *curFrame.getCloud();

		/*
		 * run icp
		 */

		rgbd::timer t;

		registration::ICPCloudPairParams cpParams;
		cpParams.errType = registration::ICP_ERR_POINT_TO_PLANE;
		cpParams.corrType = registration::ICP_CORRS_COLOR_BINNING;//color binning improves convergence but is much slower
		cpParams.max_distance = -1; //-1 = use all pts; 20110418: at least in early iters, use a large value because initial xforms are not great
		cpParams.front_can_match_back = false;
		cpParams.use_average_point_error = true; //as opposed to total
		cpParams.outlier_percentage = .1; //TODO ?
		boost::shared_ptr<registration::ICPCloudPair> cloudPair(new registration::ICPCloudPair(cpParams, curCloud, prevCloud));

		registration::ICPCombinedParams icpParams;
		icpParams.optimizer = registration::OPTIMIZER_EIGEN_LM;
		icpParams.max_icp_rounds = 40; //TODO ?
		icpParams.max_lm_rounds = 50;
		//20110419: Peter says more robust to use xform than error as convergence test
		//icpParams.min_error_frac_to_continue = 1e-3; //not very helpful; convergence is really when the xform doesn't change much
		icpParams.min_dist_to_continue = 1e-3;
		icpParams.min_rot_to_continue = 1e-3;
		registration::ICPCombined icp;
		icp.setParams(icpParams);
		icp.addCloudPair(cloudPair, 1);

		Affine3f icpXform;
		icp.runICP(icpXform, false/* verbose */);

		t.stop("run icp");

		prevXform = prevXform * icpXform;
		xforms.push_back(std::make_pair(curFrame.getImgMsg().header.stamp, prevXform));
		xf::write_transforms_to_file(xformlistFilepath.string(), xforms);

		prevFrame = curFrame;
	}

	//then use the partial map viewer

	return 0;
}
