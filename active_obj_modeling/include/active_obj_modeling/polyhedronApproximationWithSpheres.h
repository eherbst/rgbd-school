/*
 * polyhedronApproximationWithSpheres: for such things as carving free space into a voxel map
 *
 * Evan Herbst
 * 11 / 26 / 13
 */

#ifndef EX_POLYHEDRON_APPROXIMATION_WITH_SPHERES_H
#define EX_POLYHEDRON_APPROXIMATION_WITH_SPHERES_H

#include <vector>
#include <string>
#include "rgbd_util/eigen/Core"

namespace rgbd
{

struct sphereInfo
{
	bool operator < (const sphereInfo& s) const {return r > s.r;}

	rgbd::eigen::Vector3f c;
	float r;
};

/*
 * mins and maxes are of an axis-aligned bbox
 *
 * objName is used for on-disk caching
 */
std::vector<sphereInfo> approximateAABBWithSpheres(const rgbd::eigen::Vector3f& mins, const rgbd::eigen::Vector3f& maxes, const std::string& objName);

} //namespace

#endif //header
