/*
 * sceneSamplerVoxels: take samples from a scene represented as a voxel grid
 *
 * Evan Herbst
 * 2 / 15 / 12
 */

#ifndef EX_SCENE_SAMPLER_VOXELS_H
#define EX_SCENE_SAMPLER_VOXELS_H

#include <iostream>
#include "scene_rendering/voxelGridSceneInfo.h"

TODO this file won't compile; it hasn't been updated to look like the other sceneSamplers because I don't currently use it -- 20130710

/*
 * sample ID 0 is used for all unseen voxels; sample ID i > 0 corresponds to voxel index i - 1; all seen, non-free voxels are considered valid samples; sometimes this isn't what you want
 */
class sceneSamplerVoxels : public sceneSampler
{
	public:

		class sample : public sceneSample
		{
			public:

				sample(const voxelGridSceneInfo& s, const uint64_t vi) : color((*s.colors)(vi)), normal((*s.normals)(vi)), curv((*s.curvatures)(vi))
				{
					const uint16_t x = vi % s.nx, y = (vi / s.nx) % s.ny, z = vi / (s.nx * s.ny);
					pos = s.tsdf->origin + rgbd::eigen::Vector3f(x + .5, y + .5, z + .5) * s.tsdf->resolution;
				}
				virtual ~sample() {}

				virtual rgbd::eigen::Vector4f pos4f() const {return rgbd::eigen::Vector4f(pos[0], pos[1], pos[2], 1);}
				virtual rgbd::eigen::Vector4f normal4f() const {return rgbd::eigen::Vector4f(normal[0], normal[1], normal[2], 0);}
				virtual rgbd::eigen::Vector3f color3f() const {return color.cast<float>() / 255.0;}
				virtual float curvature() const {return curv;}

			private:

				rgbd::eigen::Vector3f pos;
				rgbd::eigen::Vector3uc color;
				rgbd::eigen::Vector3f normal;
				float curv;
		};

		sceneSamplerVoxels(const voxelGridSceneInfo& s) : scene(s) {}
		virtual ~sceneSamplerVoxels() {}

		virtual uint64_t numSamples() const {return scene.tsdf->voxels.num_elements();}

		/*
		 * pre: sampleValid(sampleID)
		 */
		virtual std::unique_ptr<sceneSample> getSample(const uint64_t sampleID) const
		{
			return std::unique_ptr<sample>(new sample(scene, sampleID));
		}

	private:

		const voxelGridSceneInfo& scene;
};

#endif //header
