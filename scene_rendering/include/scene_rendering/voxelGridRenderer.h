/*
 * voxelGridRenderer: render a voxel grid from many camera poses
 *
 * Evan Herbst
 * 11 / 17 / 11
 */

#ifndef EX_VOXEL_GRID_RENDERER_H
#define EX_VOXEL_GRID_RENDERER_H

#include <unordered_set>
#include <memory>
#include "rgbd_util/CameraParams.h"
#include "vrip_utils/voxelGrids.h"
#include "vrip_utils/voxelGridRendering.h"
#include "scene_rendering/sceneRenderingCache.h"

/*
 * the values we render are the totalWeights in the voxels, which should be in [0, 1]
 *
 * NOT thread-safe and not sure it's possible to make it so (because of static vars in initialization -- are there other issues?)
 */
class voxelGridRenderer : public sceneRenderingCache
{
	public:

		voxelGridRenderer(const voxelGrid<>& g, const rgbd::CameraParams& c);
		voxelGridRenderer(const voxelGrid<>& g, const std::unordered_set<voxelIndex, hashVoxelIndex>& selectedVoxelIndexSet, const rgbd::CameraParams& c);
		voxelGridRenderer(const voxelGrid<>& g, const voxelColoringFunc<OccElement>& getVoxelColor, const rgbd::CameraParams& c);
		voxelGridRenderer(const voxelGrid<>& g, const std::unordered_set<voxelIndex, hashVoxelIndex>& selectedVoxelIndexSet, const voxelColoringFunc<OccElement>& getVoxelColor, const rgbd::CameraParams& c);
		virtual ~voxelGridRenderer();

		virtual void renderAux() const;

	protected:

		void init(const voxelGrid<>& grid, const std::unordered_set<voxelIndex, hashVoxelIndex>& selectedVoxelIndexSet, const voxelColoringFunc<OccElement>& getVoxelColor);
		void initAux(const voxelGridRenderingStructures& vgrs);

		/*
		 * PIMPL, but also make dealing with glx header issues easier by keeping them in our .cpp
		 */
		struct openglData;
		std::shared_ptr<openglData> gldata;
};

#endif //header
