/*
 * testglstuff: test a voxelGridRenderer with opengl and glx
 */

#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <random>
#include <boost/format.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "rgbd_util/timer.h"
#include "rgbd_util/eigen/Geometry"
#include "rgbd_util/primesensorUtils.h"
#include "evh_util/viewSphereSampling.h"
#include "vrip_utils/vripIO.h"
#include "scene_rendering/voxelGridRenderer.h"
#include "scene_rendering/viewScoringRenderer.h"
using std::vector;
using std::ifstream;
using std::cout;
using std::endl;

int main(int argc, char **argv)
{
	ASSERT_ALWAYS(argc == 2);

	rgbd::timer u;
	std::shared_ptr<voxelGrid<>> gridptr = vrip::readRLEVRI(argv[1]);
	u.stop("read files");

	const rgbd::CameraParams camParams = primesensor::getColorCamParams(rgbd::KINECT_640_DEFAULT);

	std::mt19937 rng;
	std::uniform_real_distribution<double> dist(-1, 1);
	const vector<rgbd::eigen::Quaternionf> rots = generateWellDistributedRandomRotations([&](){return dist(rng);}, 50/*maxRotations*/, .25/*minRotSampleAngle*/);
	voxelGridRenderer renderer(*gridptr, camParams);
	viewScoringRenderer scorer(std::shared_ptr<openglContext>(), camParams);
	scorer.acquire();
	scorer.setRenderFunc([&renderer](const rgbd::eigen::Affine3f& xform){renderer.render(xform);});
	cv::Mat cvImg(camParams.yRes, camParams.xRes, CV_8UC1);
	for(unsigned int i = 0; i < rots.size(); i++)
	{
		const rgbd::eigen::Affine3f camPose = rgbd::eigen::Affine3f(rots[i]);
		scorer.render(camPose, cvImg);
		double score = 0;
		for(unsigned int j = 0; j < cvImg.rows; j++)
			for(unsigned int k = 0; k < cvImg.cols; k++)
			{
				const uint8_t c = cvImg.at<uint8_t>(j, k);
				if(c == 15) score++;
			}
		cout << score << endl;
		cv::imwrite((boost::format("gridq%1%.png") % i).str().c_str(), cvImg);
	}
	scorer.release();

	return 0;
}
