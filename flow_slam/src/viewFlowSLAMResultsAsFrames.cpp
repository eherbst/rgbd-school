/*
 * viewFlowSLAMResultsAsFrames: view all frames projected into the point of view of some particular frame
 *
 * Evan Herbst
 * 11 / 21 / 12
 */

#include <cassert>
#include <cstdint>
#include <iostream>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include "rgbd_util/xfmt.h"
#include "xforms/xforms.h"
#include "rgbd_depthmaps/conversions.h"
#include "pcl_rgbd/depth_to_cloud_lib.h"
#include "pcl_rgbd/cloudTofroPLY.h"
#include "rgbd_bag_utils/contents.h"
#include "rgbd_bag_utils/rgbdBagReader2.h"
#include "optical_flow_utils/sceneFlowIO.h"
#include "scene_rendering/viewScoringRenderer.h"
#include "scene_rendering/pointCloudRenderer.h"
using std::vector;
using std::string;
using std::cout;
using std::endl;
using boost::lexical_cast;
namespace fs = boost::filesystem;

int main(int argc, char* argv[])
{
	ASSERT_ALWAYS(argc == 7);
	const std::string type = argv[1];
	const fs::path bagFilepath = argv[2];
	const uint32_t startFrame = lexical_cast<uint32_t>(argv[3]), frameskip = lexical_cast<uint32_t>(argv[4]);
	xfmt outfmter(argv[6]); //img filepath template with %f% for frame #
	const fs::path outdir = fs::path(outfmter.set("f", 0).str()).parent_path();
	fs::create_directories(outdir);

	const rgbd::CameraParams camParams = primesensor::getColorCamParams(rgbd::KINECT_640_DEFAULT);

	string depthTopic, imgTopic, cloudTopic;
	rgbd::estimateRGBDBagSchema(bagFilepath, depthTopic, imgTopic, cloudTopic);
	rgbd::rgbdBagReader2 frameReader(bagFilepath, depthTopic, imgTopic, cloudTopic, startFrame, 1000000/* end frame */, frameskip, 0u/* num prev frames to keep */);

	if(type == "flow")
	{
		xfmt inflowFmter(argv[5]); //indir + filefmt for flow fields, with %f% for frame #

		viewScoringRenderer renderer(camParams);

		while(frameReader.readOneFrame())
		{
			fs::path inpath = inflowFmter.set("f", frameReader.getLastFrameIndex()).str();
			cout << "checking " << inpath << endl;
			if(fs::exists(inpath))
			{
				const uint32_t frameIndex = frameReader.getLastFrameIndex();
				const cv::Mat flow = readSceneFlow(inpath);

#if 1

				const sensor_msgs::ImageConstPtr imgptr = frameReader.getLastImgPtr();
				cv_bridge::CvImageConstPtr ciMsg = cv_bridge::toCvShare(imgptr, "bgr8");
				const cv::Mat& rgbImg = ciMsg->image;

				const cv::Mat_<float> depthImg = frameReader.getLastDepthImg();

				pcl::PointCloud<rgbd::pt> cloud;
				image_and_depth_to_cloud(rgbImg, depthImg, true, false, cloud, camParams);
				for(uint32_t i = 0; i < cloud.points.size(); i++)
				{
					cloud.points[i].x += flow.at<cv::Vec3f>(cloud.points[i].imgY, cloud.points[i].imgX)[0] * cloud.points[i].z / camParams.focalLength;
					cloud.points[i].y += flow.at<cv::Vec3f>(cloud.points[i].imgY, cloud.points[i].imgX)[1] * cloud.points[i].z / camParams.focalLength;
					cloud.points[i].z += flow.at<cv::Vec3f>(cloud.points[i].imgY, cloud.points[i].imgX)[2];
				}
				pointCloudRenderer samplesCache(cloud, getRenderingColorFromPointColor, 1, camParams);
				const std::function<void (const rgbd::eigen::Affine3f& xform)> renderFunc = [&samplesCache](const rgbd::eigen::Affine3f& camPose)
					{
						samplesCache.render(camPose);
					};
				renderer.acquire();
				renderer.setRenderFunc(renderFunc);
				cv::Mat_<cv::Vec3b> img(camParams.yRes, camParams.xRes);
				renderer.render(Eigen::Affine3f::Identity(), img);
				renderer.release();
				cv::imwrite(outfmter.set("f", frameIndex).str(), img);

#else

				const sensor_msgs::ImageConstPtr imgptr = frameReader.getLastImgPtr();
				cv_bridge::CvImageConstPtr ciMsg = cv_bridge::toCvShare(imgptr, "bgr8");
				const cv::Mat& img = ciMsg->image;

				ASSERT_ALWAYS(flow.size() == img.size());

				cv::Mat_<cv::Vec2f> cvFlowMat(flow.size());
				for(uint32_t i = 0; i < flow.rows; i++)
					for(uint32_t j = 0; j < flow.cols; j++)
					{
						const cv::Vec3f e = flow.at<cv::Vec3f>(i, j);
						cv::Vec2f& f = cvFlowMat(i, j);
						f[0] = j - e[0];
						f[1] = i - e[1];
					}
				cv::Mat warpedImg;
				cv::remap(img, warpedImg, cvFlowMat, cv::Mat(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
				cv::imwrite(outfmter.set("f", frameIndex).str(), warpedImg);
#endif
			}
		}
	}
	else if(type == "xf")
	{
		const fs::path xformlistFilepath = argv[5];
		const vector<std::pair<ros::Time, Eigen::Affine3f>> xforms = std::move(xf::readTransformsTextFile(xformlistFilepath.string()));

		viewScoringRenderer renderer(camParams);

		uint32_t _ = 0; //index into xforms
		while(frameReader.readOneFrame() && _ < xforms.size())
		{
			const uint32_t frameIndex = frameReader.getLastFrameIndex();
			ASSERT_ALWAYS(xforms[_].first == frameReader.getLastFrameTime());
			const Eigen::Affine3f xform = xforms[_].second;

			const sensor_msgs::ImageConstPtr imgptr = frameReader.getLastImgPtr();
			cv_bridge::CvImageConstPtr ciMsg = cv_bridge::toCvShare(imgptr, "bgr8");
			const cv::Mat& rgbImg = ciMsg->image;

			const cv::Mat_<float> depthImg = frameReader.getLastDepthImg();

			pcl::PointCloud<rgbd::pt> cloud;
			image_and_depth_to_cloud(rgbImg, depthImg, true, false, cloud, camParams);
			pointCloudRenderer samplesCache(cloud, getRenderingColorFromPointColor, 1, camParams);
			const std::function<void (const rgbd::eigen::Affine3f& xform)> renderFunc = [&samplesCache](const rgbd::eigen::Affine3f& camPose)
				{
					samplesCache.render(camPose);
				};
			renderer.acquire();
			renderer.setRenderFunc(renderFunc);
			cv::Mat_<cv::Vec3b> img(camParams.yRes, camParams.xRes);
			renderer.render(xform.inverse(), img);
			renderer.release();
			cv::imwrite(outfmter.set("f", frameIndex).str(), img);

			_++;
		}
	}
	else ASSERT_ALWAYS(false && "bad params");

	return 0;
}
