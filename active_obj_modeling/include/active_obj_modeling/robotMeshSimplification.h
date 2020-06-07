/*
 * robotMeshSimplification: reduce # triangles and otherwise make meshes easier to use
 *
 * Evan Herbst
 * 12 / 4 / 13
 */

#ifndef EX_OPENRAVE_ROBOT_MESH_SIMPLIFICATION_H
#define EX_OPENRAVE_ROBOT_MESH_SIMPLIFICATION_H

#include <cstdint>
#include <vector>
#include <tuple>
#include <openrave-core.h>
#include "vrip_utils/triangulatedMesh.h"

struct robotMeshSimplificationParams
{
	robotMeshSimplificationParams() : triDimThreshold(7e-3)
	{}

	float triDimThreshold; //minimum axis-aligned distance the triangle must span in one dimension (x, y or z) to be kept; 7e-3 works decently for barrettwam
};
/*
 * create versions with a reduced number of triangles, whether approximate meshes or just a submesh, for faster collision checking and maybe other purposes
 *
 * return: meshes; which link each mesh comes from
 */
std::tuple<std::vector<triangulatedMesh>, std::vector<uint32_t>> createSimplifiedRobotMeshes(const OpenRAVE::RobotBasePtr& robot, const robotMeshSimplificationParams& params);

#endif //header
