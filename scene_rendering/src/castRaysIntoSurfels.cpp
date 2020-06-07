/*
 * castRaysIntoSurfels: associate locations in an image plane with surfels in a cloud
 *
 * Evan Herbst
 * 7 / 30 / 10
 */

#include <cassert>
#include <cfloat> //DBL_MAX
#include <cmath> //rint()
#include <iostream>
#include <algorithm> //fill()
#include <limits>
#include "rgbd_util/eigen/LU" //inverse()
#include "rgbd_util/mathUtils.h" //sqr()
#include "rgbd_util/timer.h"
#include "scene_rendering/castRaysIntoSurfels.h"
using std::vector;
using std::cout;
using std::endl;
using rgbd::eigen::Vector2f;
using rgbd::eigen::Vector4f;
using rgbd::eigen::Vector3f;
using rgbd::eigen::Matrix;
using rgbd::eigen::Affine3f;

/*
 * project a set of surfels into a camera's (u, v, z) frame, projecting each surfel's center to a single pixel
 *
 * camXform: cam coords -> coords of surfel cloud
 *
 * return: for each pixel, id of closest surfel plus one and its distance (z in (u, v, z) ), or, if no such, id 0 and an undefined depth
 *
 * return values are indexed by (y, x), base 0
 *
 * pre: surfelIDs, surfelDepths are allocated
 */
void projectSurfelsIntoCamera(const pcl::PointCloud<rgbd::surfelPt>& surfels, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPoseWrtCloud, boost::multi_array<uint32_t, 2>& surfelIDs, boost::multi_array<float, 2>& surfelDepths)
{
	ASSERT_ALWAYS(surfelIDs.shape()[0] == camParams.yRes && surfelIDs.shape()[1] == camParams.xRes);
	ASSERT_ALWAYS(surfelDepths.shape()[0] == camParams.yRes && surfelDepths.shape()[1] == camParams.xRes);
	std::fill(surfelIDs.data(), surfelIDs.data() + surfelIDs.num_elements(), 0);
	std::fill(surfelDepths.data(), surfelDepths.data() + surfelDepths.num_elements(), std::numeric_limits<float>::max());

	const Affine3f camPoseInv(camPoseWrtCloud.inverse());

	/*
	 * project each surfel's center to a single pixel
	 */
	for(size_t i = 0; i < surfels.points.size(); i++)
	{
		const Vector4f surfelPos = rgbd::ptX2eigen<Vector4f>(surfels.points[i]);
		const Vector4f surfelPosInCamFrame = camPoseInv * surfelPos;
		const Vector2f pixelPos(camParams.focalLength * surfelPosInCamFrame.x() / surfelPosInCamFrame.z() + camParams.centerX,
										camParams.focalLength * surfelPosInCamFrame.y() / surfelPosInCamFrame.z() + camParams.centerY);
		const int x = rint(pixelPos.x()), y = rint(pixelPos.y());
		if(x >= 0 && x < camParams.xRes && y >= 0 && y < camParams.yRes)
		{
			const double depth = surfelPosInCamFrame.z();
			if(depth > 0 && depth < surfelDepths[y][x])
			{
				surfelDepths[y][x] = depth;
				surfelIDs[y][x] = i + 1;
			}
		}
	}
}

/*
 * use any scene renderer
 */
void projectSceneSamplesIntoCamera(viewScoringRenderer& renderer, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose, boost::multi_array<uint32_t, 2>& sampleIDs, boost::multi_array<float, 2>& sampleDepths)
{
	ASSERT_ALWAYS(sampleIDs.shape()[0] == camParams.yRes && sampleIDs.shape()[1] == camParams.xRes);
	ASSERT_ALWAYS(sampleDepths.shape()[0] == camParams.yRes && sampleDepths.shape()[1] == camParams.xRes);
	//create opencv wrappers around the client-allocated memory
	cv::Mat idImg(camParams.yRes, camParams.xRes, CV_8UC4, sampleIDs.data());
	cv::Mat depthImg(camParams.yRes, camParams.xRes, CV_32F, sampleDepths.data());
	renderer.acquire();
	renderer.render(camPose, idImg, depthImg);
	renderer.release();
}
/*
 * auxiliary to projectSceneSamplesIntoCameraGPU()
 *
 * edit depthsPBO to remap depths from [0, 1] to physical values
 */
void remapDepthsFromUnitRangeCUDA(const uint32_t width, const uint32_t height, const GLuint idsPBO, const GLuint depthsPBO, const float znear, const float zfar);
/*
 * write to the given pixel buffer objects
 *
 * TODO user has to manage locking the renderer
 */
void projectSceneSamplesIntoCameraGPU(viewScoringRenderer& renderer, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose, const GLuint idsPBO, const GLuint depthsPBO)
{
	renderer.renderToPixelBufferObjects(camPose, idsPBO, depthsPBO);
	/*
	 * edit depthsPBO to remap depths from [0, 1] to physical values
	 */
	remapDepthsFromUnitRangeCUDA(camParams.xRes, camParams.yRes, idsPBO, depthsPBO, renderer.zNear(), renderer.zFar());
}

/******************************************************************************************************************/

/*
 * return: for each keypt, an index into surfels or -1
 */
std::vector<int> castRaysIntoSurfels(const pcl::PointCloud<rgbd::surfelPt>& surfels, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose, const std::vector<rgbd::eigen::Vector2i>& keypts)
{
	boost::multi_array<uint32_t, 2> surfelIndices(boost::extents[camParams.yRes][camParams.xRes]);
	boost::multi_array<float, 2> surfelDepths(boost::extents[camParams.yRes][camParams.xRes]);
	projectSurfelsIntoCamera(surfels, camParams, camPose, surfelIndices, surfelDepths);

	vector<int> keypt2surfelIndex(keypts.size());
	for(unsigned int i = 0; i < keypts.size(); i++)
	{
		const int x = rint(keypts[i][0]), y = rint(keypts[i][1]);
		ASSERT_ALWAYS(x >= 0 && x < camParams.xRes && y >= 0 && y < camParams.yRes);
		if(surfelIndices[y][x] != 0) keypt2surfelIndex[i] = surfelIndices[y][x] - 1;
		else keypt2surfelIndex[i] = -1;
	}
	return keypt2surfelIndex;
}

/*
 * use the scene cloud to fill in some missing depth pixels in the single-frame depth map
 *
 * pre: depth map is uncompressed
 */
void fillInFrameDepthPixelsFromSceneCloud(rgbd_msgs::DepthMap& depth, const typename pcl::PointCloud<rgbd::surfelPt>::ConstPtr& cloud,
	const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPoseWrtMap)
{
	ASSERT_ALWAYS(depth.format == rgbd_msgs::DepthMap::format_raw);

	/*
	 * where we can, fill in unknown pixels in the original depth map using the 3-d scene map
	 *
	 * results, 20100817: we do get thousands of extra known pixels per frame here
	 */

	boost::multi_array<uint32_t, 2> surfelIndices(boost::extents[camParams.yRes][camParams.xRes]);
	boost::multi_array<float, 2> surfelDepths(boost::extents[camParams.yRes][camParams.xRes]);
	projectSurfelsIntoCamera(*cloud, camParams, camPoseWrtMap, surfelIndices, surfelDepths);

	unsigned int c = 0;
	for(unsigned int m = 0, o = 0; m < depth.height; m++)
		for(unsigned int n = 0; n < depth.width; n++, o++)
			if(depth.float_data[o] < 0 && surfelIndices[m][n] != 0)
			{
				depth.float_data[o] = surfelDepths[m][n];
				c++;
			}
}
