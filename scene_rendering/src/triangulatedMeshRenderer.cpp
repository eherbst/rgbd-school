/*
 * triangulatedMeshRenderer: render a mesh from many camera poses
 *
 * Evan Herbst
 * 11 / 16 / 11
 */

#include <cassert>
#include <iostream>
// in order to get function prototypes from gl.h and glext.h, define GL_GLEXT_PROTOTYPES first
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <opencv2/core/core.hpp>
#include "rgbd_util/timer.h"
#include "opengl_util/renderingUtils.h"
#include "scene_rendering/triangulatedMeshRenderer.h"
using std::vector;
using std::cout;
using std::endl;
using rgbd::eigen::Vector3f;

/*
 * use point's color field
 */
std::array<uint8_t, 4> getMeshVertexColorFromPointColor(const uint32_t index, const rgbd::pt& pt)
{
	const boost::array<unsigned char, 3> rgb = rgbd::unpackRGB<unsigned char>(pt.rgb);
	return std::array<uint8_t, 4>{{rgb[0], rgb[1], rgb[2], 255}};
}

/*
 * use point index
 */
std::array<uint8_t, 4> getMeshVertexColorFromID(const uint32_t index, const rgbd::pt& pt)
{
	std::array<uint8_t, 4> col;
	uint32_t* c = reinterpret_cast<uint32_t*>(col.data());
	*c = index + 1;
	return col;
}

/*
 * use triangle index
 */
std::array<uint8_t, 4> getTriangleColorFromID(const uint32_t index, const std::array<rgbd::pt, 3>& pts)
{
	std::array<uint8_t, 4> col;
	uint32_t* c = reinterpret_cast<uint32_t*>(col.data());
	*c = index + 1;
	return col;
}

/*
 * what we need in addition to what we get from viewScoringRenderer
 */
struct triangulatedMeshRenderer::openglData
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

triangulatedMeshRenderer::triangulatedMeshRenderer(const triangulatedMesh& m, const vertexColoringFunc& getVertexColor, const rgbd::CameraParams& c)
: gldata(new openglData)
{
	init(m, getVertexColor);
}
triangulatedMeshRenderer::triangulatedMeshRenderer(const triangulatedMesh& m, const triangleColoringFunc& getTriangleColor, const rgbd::CameraParams& c)
: gldata(new openglData)
{
	init(m, getTriangleColor);
}

triangulatedMeshRenderer::~triangulatedMeshRenderer()
{
	glDeleteBuffers(1, &gldata->vvboID);
	glDeleteBuffers(1, &gldata->cvboID);
	glDeleteBuffers(1, &gldata->ivboID);
	checkGLError();
}

void triangulatedMeshRenderer::init(const triangulatedMesh& mesh, const vertexColoringFunc& getVertexColor)
{
	/*
	 * init vertex buffer objects (like server-side vertex arrays)
	 */
	checkGLError();
	glGenBuffers(1, &gldata->vvboID);
	glGenBuffers(1, &gldata->cvboID);
	glGenBuffers(1, &gldata->ivboID);
	checkGLError();

	gldata->vertexArray.resize(mesh.numVertices());
	gldata->colorArray.resize(mesh.numVertices());
	for(size_t i = 0; i < mesh.numVertices(); i++)
	{
		gldata->vertexArray[i] = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(mesh.v(i));
		gldata->colorArray[i] = getVertexColor(i, mesh.v(i));
	}

	gldata->indexArray.resize(mesh.numTriangles() * 3);
	const vector<triangulatedMesh::triangle>& triangles = mesh.getTriangles();
	for(size_t i = 0, k = 0; i < mesh.numTriangles(); i++)
		for(size_t j = 0; j < 3; j++, k++)
			gldata->indexArray[k] = triangles[i].v[j];

	glBindBuffer(GL_ARRAY_BUFFER, gldata->vvboID);
	ASSERT_ALWAYS(glIsBuffer(gldata->vvboID));
	glBufferData(GL_ARRAY_BUFFER, gldata->vertexArray.size() * sizeof(rgbd::eigen::Vector3f), gldata->vertexArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ARRAY_BUFFER, gldata->cvboID);
	ASSERT_ALWAYS(glIsBuffer(gldata->cvboID));
	glBufferData(GL_ARRAY_BUFFER, gldata->colorArray.size() * sizeof(std::array<uint8_t, 4>), gldata->colorArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gldata->ivboID);
	ASSERT_ALWAYS(glIsBuffer(gldata->ivboID));
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, gldata->indexArray.size() * sizeof(uint32_t), gldata->indexArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	checkGLError();
}
void triangulatedMeshRenderer::init(const triangulatedMesh& mesh, const triangleColoringFunc& getTriangleColor)
{
	/*
	 * init vertex buffer objects (like server-side vertex arrays)
	 */
	checkGLError();
	glGenBuffers(1, &gldata->vvboID);
	glGenBuffers(1, &gldata->cvboID);
	glGenBuffers(1, &gldata->ivboID);
	checkGLError();

	gldata->vertexArray.resize(3 * mesh.numTriangles());
	gldata->colorArray.resize(3 * mesh.numTriangles());
	gldata->indexArray.resize(3 * mesh.numTriangles());
	const vector<triangulatedMesh::triangle>& triangles = mesh.getTriangles();
	for(size_t i = 0; i < mesh.numTriangles(); i++)
	{
		for(size_t j = 0; j < 3; j++)
		{
			gldata->vertexArray[i * 3 + j] = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(mesh.v(triangles[i].v[j]));
			gldata->colorArray[i * 3 + j] = getTriangleColor(i, std::array<rgbd::pt, 3>{{mesh.v(triangles[i].v[0]), mesh.v(triangles[i].v[1]), mesh.v(triangles[i].v[2])}});
			gldata->indexArray[i * 3 + j] = i * 3 + j; //TODO don't use an element index buffer when doing per-triangle coloring, for efficiency?
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, gldata->vvboID);
	ASSERT_ALWAYS(glIsBuffer(gldata->vvboID));
	glBufferData(GL_ARRAY_BUFFER, gldata->vertexArray.size() * sizeof(rgbd::eigen::Vector3f), gldata->vertexArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ARRAY_BUFFER, gldata->cvboID);
	ASSERT_ALWAYS(glIsBuffer(gldata->cvboID));
	glBufferData(GL_ARRAY_BUFFER, gldata->colorArray.size() * sizeof(std::array<uint8_t, 4>), gldata->colorArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gldata->ivboID);
	ASSERT_ALWAYS(glIsBuffer(gldata->ivboID));
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, gldata->indexArray.size() * sizeof(uint32_t), gldata->indexArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	checkGLError();
}

void triangulatedMeshRenderer::renderAux() const
{
	/*
	 * 20130403: for now, avoid 'color' interpolation when colors are sample IDs and we're using per-vertex colors by using flat shading; TODO in future maybe instead turn each triangle into a lot of points and render the mesh as a point cloud?
	 */
	glShadeModel(GL_FLAT);

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
	glDrawElements(GL_TRIANGLES, gldata->indexArray.size(), GL_UNSIGNED_INT, 0/* offset into bound buffer */);
	checkGLError();
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	checkGLError();
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
