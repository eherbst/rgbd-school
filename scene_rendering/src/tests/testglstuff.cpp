/*
 * testglstuff: test a triangulatedMeshRenderer with opengl and glx
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
#include "vrip_utils/triangulatedMesh.h"
#include "scene_rendering/triangulatedMeshRenderer.h"
#include "scene_rendering/viewScoringRenderer.h"
using std::vector;
using std::ifstream;
using std::cout;
using std::endl;

int main(int argc, char **argv)
{
	ASSERT_ALWAYS(argc == 3);

	rgbd::timer u;

	triangulatedMesh mesh;
	vector<float> vals;

	mesh.readPLY(argv[1], true);

	ifstream infile(argv[2]);
	ASSERT_ALWAYS(infile);
	unsigned int len;
	infile >> len;
	vals.resize(len);
	std::copy(std::istream_iterator<float>(infile), std::istream_iterator<float>(), vals.begin());
	infile.close();

	ASSERT_ALWAYS(mesh.numVertices() == vals.size());
	u.stop("read files");

	const rgbd::CameraParams camParams = primesensor::getColorCamParams(rgbd::KINECT_640_DEFAULT);

	std::mt19937 rng;
	std::uniform_real_distribution<double> dist(-1, 1);
	const vector<rgbd::eigen::Quaternionf> rots = generateWellDistributedRandomRotations([&](){return dist(rng);}, 50/*maxRotations*/, .25/*minRotSampleAngle*/);
	meshVertexColorerFromScalar meshColorer(vals);
	const triangulatedMeshRenderer::vertexColoringFunc coloringFunc = [&meshColorer](const uint32_t index, const rgbd::pt& pt){return meshColorer.getVertexColorFromScalar(index, pt);};
	triangulatedMeshRenderer renderer(mesh, coloringFunc, camParams);
	viewScoringRenderer scorer(std::shared_ptr<openglContext>(), camParams);
	scorer.acquire();
	scorer.setRenderFunc([&renderer](const rgbd::eigen::Affine3f& xform){renderer.render(xform);});
	cv::Mat cvImg(2048, 2048, CV_8UC3);//camParams.yRes, camParams.xRes, CV_8UC3);
	for(unsigned int i = 0; i < rots.size(); i++)
	{
		rgbd::timer t;
		const rgbd::eigen::Affine3f camPose = rgbd::eigen::Affine3f(rots[i]);
#if 1
		vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>> camPoses(29);
		std::fill(camPoses.begin(), camPoses.end(), camPose);
		//const float score2 = scorer.renderAndScore(camPose);
		const vector<float> scores2 = scorer.renderAndScore(camPoses);
		cout << "avg " << scores2[0] << endl;
#endif
#if 0
		scorer.render(camPose, cvImg);
		double score = 0;
		for(unsigned int j = 0; j < cvImg.rows; j++)
			for(unsigned int k = 0; k < cvImg.cols; k++)
			{
				const uint8_t c = cvImg.at<uint8_t>(j, k);
				score += c;
			}
		cout << "score " << score << endl;
		cv::imwrite((boost::format("meshq%1%.png") % i).str().c_str(), cvImg);
#endif
		t.stop("run whole loop, one pose");
		ASSERT_ALWAYS(false);
	}
	scorer.release();

	return 0;
}
