/*
 * collisionChecking: we seem to be able to do it much more quickly ourselves than openrave can
 *
 * Evan Herbst
 * 12 / 20 / 13
 */

#ifndef EX_ACTIVE_VISION_COLLISION_CHECKING_H
#define EX_ACTIVE_VISION_COLLISION_CHECKING_H

#include <cstdint>
#include <string>
#include <memory>
#include <openrave-core.h>
#include "rgbd_util/eigen/Geometry"
#include "rgbd_util/eigen/StdVector"
#include "vrip_utils/triangulatedMesh.h"
#include "active_vision_common/robotSpec.h"

struct precomputedEnvCollisionData; //something on the order of an aabb tree over the environment

std::shared_ptr<precomputedEnvCollisionData> precomputeForCollisionChecking(const std::vector<std::shared_ptr<triangulatedMesh>>& envMeshes,
	const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& envMeshPosesWrtRaveWorld);

/*
 * return whether each robot configuration puts the robot in collision with the environment
 *
 * output is undefined if a configuration is the empty vector
 *
 * use raveEnvs for multithreading
 *
 * robotLinkMeshLinkIndices: which link each mesh comes from
 */
std::vector<uint8_t> checkRAVERobotCollisions(const std::vector<OpenRAVE::EnvironmentBasePtr>& raveEnvs, const std::string& robotName, robotSpec& robotInterface, const std::shared_ptr<precomputedEnvCollisionData>& envData,
	const std::vector<triangulatedMesh>& robotLinkMeshes, const std::vector<uint32_t>& robotLinkMeshLinkIndices, const std::vector<std::vector<OpenRAVE::dReal>>& configurations);
std::vector<uint8_t> checkRAVERobotCollisions(const std::vector<OpenRAVE::EnvironmentBasePtr>& raveEnvs, const std::string& robotName, robotSpec& robotInterface, const std::vector<std::shared_ptr<triangulatedMesh>>& envMeshes,
	const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& envMeshPosesWrtRaveWorld, const std::vector<triangulatedMesh>& robotLinkMeshes, const std::vector<uint32_t>& robotLinkMeshLinkIndices,
	const std::vector<std::vector<OpenRAVE::dReal>>& configurations);

#endif //header
