/*
 * voxelGridSceneInfo: a 3-d scene represented as voxels
 *
 * Evan Herbst
 * 2 / 15 / 12
 */

#include <cassert>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "rgbd_util/yamlUtils.h"
#include "vrip_utils/vripIO.h"
#include "vrip_utils/voxelGridIO.h"
#include "scene_rendering/voxelGridSceneInfo.h"
using std::ifstream;

void voxelGridSceneInfo::load(const fs::path& scenefilepath)
{
	const fs::path parentDir = scenefilepath.parent_path();
	fs::path tsdfGridPath, normalsGridPath, colorsGridPath;
{
	ifstream infile(scenefilepath.string().c_str());
	ASSERT_ALWAYS(infile);
	YAML::Parser parser(infile);
	YAML::Node doc;
	parser.GetNextDocument(doc);
	getOptionalRelativePath(doc, "tsdfGrid", tsdfGridPath, parentDir);
	getOptionalRelativePath(doc, "normalsGrid", normalsGridPath, parentDir);
	getOptionalRelativePath(doc, "colorsGrid", colorsGridPath, parentDir);
}
	ASSERT_ALWAYS(!tsdfGridPath.empty());
	ASSERT_ALWAYS(!normalsGridPath.empty());
	ASSERT_ALWAYS(!colorsGridPath.empty());

	tsdf = vrip::readRLEVRI(tsdfGridPath);
	nx = tsdf->voxels.shape()[2];
	ny = tsdf->voxels.shape()[1];
	nz = tsdf->voxels.shape()[0];
	normals = voxels::readVoxelNormalsGrid(normalsGridPath);
	colors = voxels::readVoxelColorsGrid(colorsGridPath);
	for(size_t i = 0; i < 3; i++)
	{
		ASSERT_ALWAYS(normals->voxels.shape()[i] == tsdf->voxels.shape()[i]);
		ASSERT_ALWAYS(colors->voxels.shape()[i] == tsdf->voxels.shape()[i]);
	}

	/*
	 * TODO get curvatures from normals? from pcl on voxel centers?
	 */
	curvatures.reset(new voxelGrid<float>);
	curvatures->voxels.resize(boost::extents[nz][ny][nx]);
	std::fill(curvatures->voxels.data(), curvatures->voxels.data() + curvatures->voxels.num_elements(), 1.0 / 0); //raise the chance we'll notice something wrong if we try to use them
}
