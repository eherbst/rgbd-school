/*
 * sceneSamplerMesh: take scene samples from the vertices of a mesh
 *
 * Evan Herbst
 * 3 / 28 / 13
 */

#ifndef EX_SCENE_SAMPLER_MESH_H
#define EX_SCENE_SAMPLER_MESH_H

#include <cassert>
#include <cstdint>
#include <memory>
#include "pcl_rgbd/pointTypes.h"
#include "vrip_utils/triangulatedMesh.h"
#include "scene_rendering/sceneSampler.h"

/*
 * use triangles as samples
 */
class sceneSamplerMesh : public sceneSampler
{
	public:

		sceneSamplerMesh(const std::shared_ptr<triangulatedMesh>& m) : mesh(m) {ASSERT_ALWAYS(mesh);}
		virtual ~sceneSamplerMesh() {}

		virtual uint64_t numSamples() const {return mesh->numTriangles();}

		/*
		 * output arrays will be allocated if nec
		 */
		virtual void getPerPixelInfo(const boost::multi_array<uint32_t, 2>& sampleIDsPerPixel, boost::multi_array<rgbd::eigen::Vector3f, 2>& expectedNormalsPerPixel, boost::multi_array<boost::array<uint8_t, 3>, 2>& expectedColsPerPixel) const
		{
			expectedNormalsPerPixel.resize(boost::extents[sampleIDsPerPixel.shape()[0]][sampleIDsPerPixel.shape()[1]]);
			expectedColsPerPixel.resize(boost::extents[sampleIDsPerPixel.shape()[0]][sampleIDsPerPixel.shape()[1]]);
			const auto& triangles = mesh->getTriangles();
			for(size_t i = 0; i < sampleIDsPerPixel.shape()[0]; i++)
				for(size_t j = 0; j < sampleIDsPerPixel.shape()[1]; j++)
					if(sampleIDsPerPixel[i][j] > 0)
					{
						expectedNormalsPerPixel[i][j] = mesh->getTriangleNormal(sampleIDsPerPixel[i][j] - 1);
						const rgbd::pt p0 = mesh->v(triangles[sampleIDsPerPixel[i][j] - 1].v[0]), p1 = mesh->v(triangles[sampleIDsPerPixel[i][j] - 1].v[1]), p2 = mesh->v(triangles[sampleIDsPerPixel[i][j] - 1].v[2]);
						const rgbd::eigen::Vector3f rgb0 = rgbd::unpackRGB2eigen(p0.rgb), rgb1 = rgbd::unpackRGB2eigen(p1.rgb), rgb2 = rgbd::unpackRGB2eigen(p2.rgb), rgbAvg = (rgb0 + rgb1 + rgb2) / 3;
						expectedColsPerPixel[i][j] = boost::array<uint8_t, 3>{{(uint8_t)(255 * rgbAvg[0]), (uint8_t)(255 * rgbAvg[1]), (uint8_t)(255 * rgbAvg[2])}};
					}
		}
		virtual void getSamplePts(std::vector<rgbd::eigen::Vector3f>& samplePts) const
		{
			ASSERT_ALWAYS(false && "unimplemented");
		}
		virtual void getSampleNormals(std::vector<rgbd::eigen::Vector3f>& sampleNormals) const
		{
			ASSERT_ALWAYS(false && "unimplemented");
		}
		virtual void getSampleCols(std::vector<boost::array<uint8_t, 3>>& sampleCols) const
		{
			const auto& triangles = mesh->getTriangles();
			sampleCols.resize(mesh->numTriangles());
			for(size_t i = 0; i < mesh->numTriangles(); i++)
			{
				const rgbd::pt p0 = mesh->v(triangles[i].v[0]), p1 = mesh->v(triangles[i].v[1]), p2 = mesh->v(triangles[i].v[2]);
				const rgbd::eigen::Vector3f rgb0 = rgbd::unpackRGB2eigen(p0.rgb), rgb1 = rgbd::unpackRGB2eigen(p1.rgb), rgb2 = rgbd::unpackRGB2eigen(p2.rgb), rgbAvg = (rgb0 + rgb1 + rgb2) / 3;
				sampleCols[i] = boost::array<uint8_t, 3>{{(uint8_t)(255 * rgbAvg[0]), (uint8_t)(255 * rgbAvg[1]), (uint8_t)(255 * rgbAvg[2])}};
			}
		}

	private:

		std::shared_ptr<triangulatedMesh> mesh;
};

#endif //header
