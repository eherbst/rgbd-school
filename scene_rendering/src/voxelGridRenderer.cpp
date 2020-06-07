/*
 * voxelGridRenderer: render a voxel grid from many camera poses
 *
 * Evan Herbst
 * 11 / 17 / 11
 */

#include <cassert>
#include <array>
#include <iostream>
// in order to get function prototypes from gl.h and glext.h, define GL_GLEXT_PROTOTYPES first
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <opencv2/core/core.hpp>
#include "rgbd_util/assert.h"
#include "rgbd_util/timer.h"
#include "vrip_utils/voxelGridRendering.h"
#include "opengl_util/renderingUtils.h"
#include "scene_rendering/voxelGridRenderer.h"
using std::vector;
using std::cout;
using std::endl;
using rgbd::eigen::Vector3f;

/*
 * what we need in addition to what we get from viewScoringRenderer
 */
struct voxelGridRenderer::openglData
{
	/*
	 * opengl stuff
	 */
	GLuint vvboID; //vertex buffer object for vertices
	GLuint cvboID; //vertex buffer object for colors
	GLuint ivboID; //vertex buffer object for indices

	vector<rgbd::eigen::Vector3f> vertexArray;
	vector<std::array<uint8_t, 4>> colorArray;
	vector<uint32_t> indexArray;
};

voxelGridRenderer::voxelGridRenderer(const voxelGrid<>& g, const rgbd::CameraParams& c)
: gldata(new openglData)
{
	const std::unordered_set<voxelIndex, hashVoxelIndex> selectedVoxelIndexSet = std::move(selectAllNonfreeVoxels(g));
	const auto getVoxelColor = [](const OccElement& e, const std::array<uint32_t, 3>& i, const std::array<uint32_t, 3>& n){return getVoxelColorFromValueForScoring(e);};
	init(g, selectedVoxelIndexSet, getVoxelColor);
}
voxelGridRenderer::voxelGridRenderer(const voxelGrid<>& g, const std::unordered_set<voxelIndex, hashVoxelIndex>& selectedVoxelIndexSet, const rgbd::CameraParams& c)
: gldata(new openglData)
{
	const auto getVoxelColor = [](const OccElement& e, const std::array<uint32_t, 3>& i, const std::array<uint32_t, 3>& n){return getVoxelColorFromValueForScoring(e);};
	init(g, selectedVoxelIndexSet, getVoxelColor);
}
voxelGridRenderer::voxelGridRenderer(const voxelGrid<>& g, const voxelColoringFunc<OccElement>& getVoxelColor, const rgbd::CameraParams& c)
: gldata(new openglData)
{
	const std::unordered_set<voxelIndex, hashVoxelIndex> selectedVoxelIndexSet = std::move(selectAllNonfreeVoxels(g));
	init(g, selectedVoxelIndexSet, getVoxelColor);
}
voxelGridRenderer::voxelGridRenderer(const voxelGrid<>& g, const std::unordered_set<voxelIndex, hashVoxelIndex>& selectedVoxelIndexSet, const voxelColoringFunc<OccElement>& getVoxelColor, const rgbd::CameraParams& c)
: gldata(new openglData)
{
	init(g, selectedVoxelIndexSet, getVoxelColor);
}

voxelGridRenderer::~voxelGridRenderer()
{
	glDeleteBuffers(1, &gldata->vvboID);
	glDeleteBuffers(1, &gldata->cvboID);
	glDeleteBuffers(1, &gldata->ivboID);
	checkGLError();
}

void voxelGridRenderer::init(const voxelGrid<>& grid, const std::unordered_set<voxelIndex, hashVoxelIndex>& selectedVoxelIndexSet, const voxelColoringFunc<OccElement>& getVoxelColor)
{
	const voxelGridRenderingStructures vgrs = std::move(initRenderingInfoSelectedVoxels(grid, selectedVoxelIndexSet, getVoxelColor));
	initAux(vgrs);
}
void voxelGridRenderer::initAux(const voxelGridRenderingStructures& vgrs)
{
	/*
	 * init vertex buffer objects (like server-side vertex arrays)
	 */
	glGenBuffers(1, &gldata->vvboID);
	glGenBuffers(1, &gldata->cvboID);
	glGenBuffers(1, &gldata->ivboID);
	checkGLError();
	ASSERT_ALWAYS(glIsBuffer(gldata->vvboID));
	ASSERT_ALWAYS(glIsBuffer(gldata->cvboID));
	ASSERT_ALWAYS(glIsBuffer(gldata->ivboID));
	checkGLError();

	gldata->vertexArray = std::move(vgrs.vertexArray);
	gldata->colorArray = std::move(vgrs.colorArray);
	gldata->indexArray = std::move(vgrs.indexArray);

	glBindBuffer(GL_ARRAY_BUFFER, gldata->vvboID);
	glBufferData(GL_ARRAY_BUFFER, gldata->vertexArray.size() * sizeof(rgbd::eigen::Vector3f), gldata->vertexArray.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, gldata->cvboID);
	glBufferData(GL_ARRAY_BUFFER, gldata->colorArray.size() * sizeof(std::array<uint8_t, 4>), gldata->colorArray.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gldata->ivboID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, gldata->indexArray.size() * sizeof(uint32_t), gldata->indexArray.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void voxelGridRenderer::renderAux() const
{
#if 0 //for testing: all light blue
	glClearColor ( 0, 0.5, 1, 1 );
	glClear ( GL_COLOR_BUFFER_BIT );
#else //draw grid

	glBindBuffer(GL_ARRAY_BUFFER, gldata->vvboID);
	glVertexPointer(3, GL_FLOAT, 0, 0/* offset into bound buffer */);
	checkGLError();
	glBindBuffer(GL_ARRAY_BUFFER, gldata->cvboID);
	glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0/* offset into bound buffer */);
	checkGLError();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gldata->ivboID);
	checkGLError();
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	checkGLError();
	glDrawElements(GL_QUADS, gldata->indexArray.size(), GL_UNSIGNED_INT, 0/* offset into bound buffer */);
	checkGLError();
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	checkGLError();
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

#endif
}
