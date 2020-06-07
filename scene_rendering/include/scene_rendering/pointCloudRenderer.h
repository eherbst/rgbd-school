/*
 * pointCloudRenderer: render a point cloud from many camera poses
 *
 * Evan Herbst
 * 2 / 20 / 12
 */

#ifndef EX_POINT_CLOUD_RENDERER_H
#define EX_POINT_CLOUD_RENDERER_H

#include <array>
#include <vector>
#include <functional>
#include <pcl/point_cloud.h>
#include "rgbd_util/CameraParams.h"
#include "pcl_rgbd/pointTypes.h"
#include "scene_rendering/sceneRenderingCache.h"

/*
 * use point's color field
 */
std::array<uint8_t, 4> getRenderingColorFromPointColor(const uint32_t index, const rgbd::pt& pt);
/*
 * use pt index
 */
std::array<uint8_t, 4> getPointColorFromID(const uint32_t index, const rgbd::pt& pt);

class pointCloudRenderer : public sceneRenderingCache
{
	public:

		/*
		 * index: into whatever cloud the point came from
		 */
		typedef std::function<std::array<uint8_t, 4> (const uint32_t index, const rgbd::pt& pt)> pointColoringFunc;

		/*
		 * pre: cloud has normals set
		 */
		pointCloudRenderer(const pcl::PointCloud<rgbd::pt>& cloud, const uint32_t ptSize, const rgbd::CameraParams& c);
		pointCloudRenderer(const pcl::PointCloud<rgbd::pt>& cloud, const pointColoringFunc& getPtColor, const uint32_t ptSize, const rgbd::CameraParams& c);
		virtual ~pointCloudRenderer();

		void reloadColors(const pcl::PointCloud<rgbd::pt>& cloud, const pointColoringFunc& getPtColor);

		virtual void renderAux() const;

	protected:

		void init(const pcl::PointCloud<rgbd::pt>& cloud, const pointColoringFunc& getPtColor);

		/*
		 * PIMPL, but also make dealing with glx header issues easier by keeping them in our .cpp
		 */
		struct openglData;
		std::shared_ptr<openglData> gldata;
};

#endif //header
