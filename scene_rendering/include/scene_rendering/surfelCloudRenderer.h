/*
 * surfelCloudRenderer: render a surfel cloud from many camera poses
 *
 * Evan Herbst
 * 12 / 8 / 11
 */

#ifndef EX_SURFEL_CLOUD_RENDERER_H
#define EX_SURFEL_CLOUD_RENDERER_H

#include <array>
#include <vector>
#include <functional>
#include <pcl/point_cloud.h>
#include "rgbd_util/CameraParams.h"
#include "pcl_rgbd/pointTypes.h"
#include "scene_rendering/sceneRenderingCache.h"

/*
 * use surfel's color field
 */
std::array<uint8_t, 4> getSurfelColorFromPointColor(const uint32_t index, const rgbd::surfelPt& pt);
/*
 * use surfel index
 */
std::array<uint8_t, 4> getSurfelColFromID(const uint32_t index, const rgbd::surfelPt& pt);

class surfelCloudRenderer : public sceneRenderingCache
{
	public:

		/*
		 * index: into whatever cloud the point came from
		 */
		typedef std::function<std::array<uint8_t, 4> (const uint32_t index, const rgbd::surfelPt& pt)> surfelColoringFunc;

		/*
		 * pre: cloud has normals set
		 */
		surfelCloudRenderer(const pcl::PointCloud<rgbd::surfelPt>& cloud, const rgbd::CameraParams& c);
		surfelCloudRenderer(const pcl::PointCloud<rgbd::surfelPt>& cloud, const surfelColoringFunc& getSurfelColor, const rgbd::CameraParams& c);
		virtual ~surfelCloudRenderer();

		void reloadColors(const pcl::PointCloud<rgbd::surfelPt>& cloud, const surfelColoringFunc& getSurfelColor);

		virtual void renderAux() const;

	protected:

		void init(const pcl::PointCloud<rgbd::surfelPt>& cloud, const surfelColoringFunc& getSurfelColor);

		/*
		 * PIMPL, but also make dealing with glx header issues easier by keeping them in our .cpp
		 */
		struct openglData;
		std::shared_ptr<openglData> gldata;
};

#endif //header
