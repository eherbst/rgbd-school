/*
 * triangulatedMeshRenderer: render a mesh from many camera poses
 *
 * Evan Herbst
 * 11 / 16 / 11
 */

#ifndef EX_TRIANGULATED_MESH_RENDERER_H
#define EX_TRIANGULATED_MESH_RENDERER_H

#include <array>
#include <vector>
#include "rgbd_util/CameraParams.h"
#include "pcl_rgbd/pointTypes.h"
#include "vrip_utils/triangulatedMesh.h"
#include "scene_rendering/sceneRenderingCache.h"

/*
 * use point's color field
 */
std::array<uint8_t, 4> getMeshVertexColorFromPointColor(const uint32_t index, const rgbd::pt& pt);
/*
 * use point index
 */
std::array<uint8_t, 4> getMeshVertexColorFromID(const uint32_t index, const rgbd::pt& pt);
/*
 * use a single continuous value in [0, 1] at each vertex
 */
class meshVertexColorerFromScalar
{
	public:

		meshVertexColorerFromScalar(const std::vector<float>& v) : vals(v) {}

		std::array<uint8_t, 4> getVertexColorFromScalar(const uint32_t index, const rgbd::pt& pt)
		{
			return std::array<uint8_t, 4>{{255, (uint8_t)(1 + rint(254 * vals[index])), 0, 255}}; //leave green=0 to mean free space
		}

	private:

		const std::vector<float>& vals;
};

/*
 * use triangle index
 */
std::array<uint8_t, 4> getTriangleColorFromID(const uint32_t index, const std::array<rgbd::pt, 3>& pts);


class triangulatedMeshRenderer : public sceneRenderingCache
{
	public:

		/*
		 * index: into whatever cloud the point came from
		 */
		typedef std::function<std::array<uint8_t, 4> (const uint32_t index, const rgbd::pt& pt)> vertexColoringFunc;
		/*
		 * index: into our mesh's list of triangles
		 */
		typedef std::function<std::array<uint8_t, 4> (const uint32_t index, const std::array<rgbd::pt, 3>& pts)> triangleColoringFunc;

		triangulatedMeshRenderer(const triangulatedMesh& mesh, const vertexColoringFunc& getVertexColor, const rgbd::CameraParams& c);
		triangulatedMeshRenderer(const triangulatedMesh& mesh, const triangleColoringFunc& getTriangleColor, const rgbd::CameraParams& c);
		virtual ~triangulatedMeshRenderer();

		virtual void renderAux() const;

	protected:

		void init(const triangulatedMesh& m, const vertexColoringFunc& getVertexColor);
		void init(const triangulatedMesh& m, const triangleColoringFunc& getTriangleColor);

		/*
		 * PIMPL, but also make dealing with glx header issues easier by keeping them in our .cpp
		 */
		struct openglData;
		std::shared_ptr<openglData> gldata;
};

#endif //header
