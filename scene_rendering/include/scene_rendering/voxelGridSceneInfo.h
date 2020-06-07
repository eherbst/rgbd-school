/*
 * voxelGridSceneInfo: a 3-d scene represented as voxels
 *
 * Evan Herbst
 * 2 / 15 / 12
 */

#ifndef EX_VOXEL_GRID_SCENE_INFO_H
#define EX_VOXEL_GRID_SCENE_INFO_H

#include <cstdint>
#include <memory>
#include <boost/filesystem/path.hpp>
#include "rgbd_util/eigen/Geometry"
#include "vrip_utils/voxelGrids.h"
namespace fs = boost::filesystem;

class voxelGridSceneInfo
{
	public:

		void load(const fs::path& scenefilepath);

		std::shared_ptr<voxelGrid<>> tsdf;
		std::shared_ptr<voxelGrid<rgbd::eigen::Vector3f>> normals;
		std::shared_ptr<voxelGrid<rgbd::eigen::Vector3uc>> colors;
		std::shared_ptr<voxelGrid<float>> curvatures;
		uint16_t nx, ny, nz; //grid dimensions
};

#endif //header
