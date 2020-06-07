/*
 * castRaysIntoSurfels: associate locations in an image plane with surfels in a cloud
 *
 * Evan Herbst
 * 7 / 30 / 10
 */

#ifndef EX_CAST_RAYS_INTO_SURFELS_H
#define EX_CAST_RAYS_INTO_SURFELS_H

#include <vector>
#include <tuple>
#include <utility>
#include <boost/multi_array.hpp>
#include <pcl/point_cloud.h>
#include "rgbd_msgs/DepthMap.h"
#include "rgbd_util/CameraParams.h"
#include "rgbd_util/eigen/Geometry"
#include "pcl_rgbd/pointTypes.h"
#include "scene_rendering/viewScoringRenderer.h"
#include "scene_rendering/sceneSampler.h"

/*
 * project a set of surfels into a camera's (u, v, z) frame, projecting each surfel's center to a single pixel
 *
 * return: for each pixel, id of closest surfel plus one and its distance (z in (u, v, z) ), or, if no such, id 0 and an undefined depth
 *
 * return values are indexed by (y, x), base 0
 *
 * pre: surfelIDs, surfelDepths are allocated
 */
void projectSurfelsIntoCamera(const pcl::PointCloud<rgbd::surfelPt>& surfels, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPoseWrtCloud, boost::multi_array<uint32_t, 2>& surfelIDs, boost::multi_array<float, 2>& surfelDepths);

/*
 * use any scene renderer
 *
 * return: for each pixel, id of closest sample plus one and its distance (z in (u, v, z) ), or, if no such, id 0 and an undefined depth
 *
 * return values are indexed by (y, x), base 0
 *
 * pre: sampleIDs, sampleDepths are allocated
 */
void projectSceneSamplesIntoCamera(viewScoringRenderer& renderer, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose, boost::multi_array<uint32_t, 2>& sampleIDs, boost::multi_array<float, 2>& sampleDepths);
/*
 * write to the given pixel buffer objects
 */
void projectSceneSamplesIntoCameraGPU(viewScoringRenderer& renderer, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose, const GLuint idsPBO, const GLuint depthsPBO);

/*
 * project a set of 3-d points into a camera's frame
 *
 * return: for each pixel, id of closest point plus one and its distance (z in (u, v, z) ), or, if no such, id 0 and an undefined depth
 *
 * ForwardRangeT's value type should be convertible to Vector3f
 *
 * pre: ptIDs, ptDepths are allocated
 */
template <typename ForwardRangeT>
void projectPointsIntoCamera(const ForwardRangeT& pts3d, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose, boost::multi_array<uint32_t, 2>& ptIDs, boost::multi_array<float, 2>& ptDepths);

/*
 * project a set of points into a camera's (u, v, z) frame
 *
 * camXform: cam coords -> coords of surfel cloud
 *
 * return: for each surfel, image (u, v, z) in (pixels, pixels, m) in the camera's frame
 * (may not be in image bounds; check if you care)
 */
template <typename PointT>
std::tuple<std::vector<rgbd::eigen::Vector2i>, std::vector<float>> projectAllCloudPointsIntoCamera(const pcl::PointCloud<PointT>& cloud, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose);
/*
 * project a set of 3-d points into a camera's frame
 *
 * camXform: cam coords -> coords of the cloud the pts3d came from
 *
 * return: for each pt, image (u, v, z) in (pixels, pixels, m) in the camera's frame
 * (may not be in image bounds; check if you care)
 *
 * ForwardRangeT's value type should be convertible to Vector3f
 */
template <typename ForwardRangeT>
std::tuple<std::vector<rgbd::eigen::Vector2i>, std::vector<float>> projectAllPointsIntoCamera(const ForwardRangeT& pts3d, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose);

/******************************************************************************************************************/

/*
 * return: for each keypt, an index into surfels or -1
 */
std::vector<int> castRaysIntoSurfels(const pcl::PointCloud<rgbd::surfelPt>& surfels, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose, const std::vector<rgbd::eigen::Vector2i>& keypts);

/*
 * use the scene cloud to fill in some missing depth pixels in the single-frame depth map
 *
 * pre: depth map is uncompressed
 */
void fillInFrameDepthPixelsFromSceneCloud(rgbd_msgs::DepthMap& depth, const pcl::PointCloud<rgbd::surfelPt>::ConstPtr& cloud, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPoseWrtMap);

#include "castRaysIntoSurfels.ipp"

#endif //header
