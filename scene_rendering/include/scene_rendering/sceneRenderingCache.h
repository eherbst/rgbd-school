/*
 * sceneRenderingCache: vertex buffer objects and such for rendering some representation of a scene
 *
 * Evan Herbst
 * 2 / 20 / 12
 */

#ifndef EX_SCENE_RENDERING_CACHE_H
#define EX_SCENE_RENDERING_CACHE_H

#include "rgbd_util/eigen/Geometry"

class sceneRenderingCache
{
	public:

		virtual ~sceneRenderingCache() {}

		void render(const rgbd::eigen::Affine3f& camPose) const;

	protected:

		virtual void renderAux() const = 0;
};

#endif //header
