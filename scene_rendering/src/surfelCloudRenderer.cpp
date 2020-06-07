/*
 * surfelCloudRenderer: render a surfel cloud from many camera poses
 *
 * Evan Herbst
 * 12 / 8 / 11
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
#include "scene_rendering/surfelCloudRenderer.h"
using std::vector;
using std::cout;
using std::endl;
using rgbd::eigen::Vector3f;

/*
 * use surfel's color field
 */
std::array<uint8_t, 4> getSurfelColorFromPointColor(const uint32_t index, const rgbd::surfelPt& pt)
{
	const boost::array<unsigned char, 3> rgb = rgbd::unpackRGB<unsigned char>(pt.rgb);
	return std::array<uint8_t, 4>{{rgb[0], rgb[1], rgb[2], 255}};
}
/*
 * use surfel index
 */
std::array<uint8_t, 4> getSurfelColFromID(const uint32_t index, const rgbd::surfelPt& pt)
{
	std::array<uint8_t, 4> col;
	uint32_t* c = reinterpret_cast<uint32_t*>(col.data());
	*c = index + 1;
	return col;
}

/*
 * what we need in addition to what we get from viewScoringRenderer
 */
struct surfelCloudRenderer::openglData
{
	static const size_t trianglesPerSurfel = 6;

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

/*
 * pre: cloud has normals set
 */
surfelCloudRenderer::surfelCloudRenderer(const pcl::PointCloud<rgbd::surfelPt>& cloud, const rgbd::CameraParams& c)
: gldata(new openglData)
{
	init(cloud, getSurfelColorFromPointColor);
}
surfelCloudRenderer::surfelCloudRenderer(const pcl::PointCloud<rgbd::surfelPt>& cloud, const surfelColoringFunc& getSurfelColor, const rgbd::CameraParams& c)
: gldata(new openglData)
{
	init(cloud, getSurfelColor);
}

surfelCloudRenderer::~surfelCloudRenderer()
{
	glDeleteBuffers(1, &gldata->vvboID);
	glDeleteBuffers(1, &gldata->cvboID);
	glDeleteBuffers(1, &gldata->ivboID);
	checkGLError();
}

void surfelCloudRenderer::init(const pcl::PointCloud<rgbd::surfelPt>& cloud, const surfelColoringFunc& getSurfelColor)
{
	/*
	 * init vertex buffer objects (like server-side vertex arrays)
	 */
	checkGLError();
	glGenBuffers(1, &gldata->vvboID);
	glGenBuffers(1, &gldata->cvboID);
	glGenBuffers(1, &gldata->ivboID);
	checkGLError();

	const float radiusExpandFactor = 1.5; //make the triangles bigger than the actual surfels to help w/ occluding other surfaces

	/*
	 * for each surfel add a circle in its plane
	 *
	 * 20111210: tried adding out-of-plane circles so it was more like a sphere; didn't improve lack of occluding surfaces in some places
	 */
	gldata->vertexArray.resize(cloud.points.size() * (gldata->trianglesPerSurfel + 1));
	gldata->colorArray.resize(gldata->vertexArray.size());
	for(unsigned int i = 0; i < cloud.points.size(); i++)
	{
		const size_t istart = i * (gldata->trianglesPerSurfel + 1);
		gldata->vertexArray[istart] = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(cloud.points[i]);
		const Vector3f normal = rgbd::ptNormal2eigen<rgbd::eigen::Vector3f>(cloud.points[i]);
		Vector3f axis1 = Vector3f::UnitX(); if(axis1.dot(normal) > .9) axis1 = Vector3f::UnitY();
		axis1 = (axis1 - axis1.dot(normal) * normal).normalized();
		const Vector3f axis2 = normal.cross(axis1);
		const float radius = cloud.points[i].radius * radiusExpandFactor;
		ASSERT_ALWAYS(radius > 1e-5);
		for(unsigned int j = 0; j < gldata->trianglesPerSurfel; j++)
		{
			const double theta = j * 2 * M_PI / gldata->trianglesPerSurfel;
			gldata->vertexArray[istart + j + 1] = gldata->vertexArray[istart] + radius * (cos(theta) * axis1 + sin(theta) * axis2);
		}
		for(unsigned int j = 0; j < gldata->trianglesPerSurfel + 1; j++)
			gldata->colorArray[istart + j] = getSurfelColor(i, cloud.points[i]);
	}

	gldata->indexArray.resize(cloud.points.size() * (gldata->trianglesPerSurfel * 3));
	for(unsigned int i = 0, k = 0; i < cloud.points.size(); i++)
		for(size_t l = 0; l < gldata->trianglesPerSurfel; l++)
		{
			gldata->indexArray[k++] = (i * (gldata->trianglesPerSurfel + 1)) + 0;
			gldata->indexArray[k++] = (i * (gldata->trianglesPerSurfel + 1)) + 1 + l;
			gldata->indexArray[k++] = (i * (gldata->trianglesPerSurfel + 1)) + 1 + (1 + l) % gldata->trianglesPerSurfel;
		}

	glBindBuffer(GL_ARRAY_BUFFER, gldata->vvboID);
	checkGLError();
	ASSERT_ALWAYS(glIsBuffer(gldata->vvboID));
	checkGLError();
	glBufferData(GL_ARRAY_BUFFER, gldata->vertexArray.size() * sizeof(rgbd::eigen::Vector3f), gldata->vertexArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ARRAY_BUFFER, gldata->cvboID);
	checkGLError();
	ASSERT_ALWAYS(glIsBuffer(gldata->cvboID));
	checkGLError();
	glBufferData(GL_ARRAY_BUFFER, gldata->colorArray.size() * sizeof(std::array<uint8_t, 4>), gldata->colorArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gldata->ivboID);
	checkGLError();
	ASSERT_ALWAYS(glIsBuffer(gldata->ivboID));
	checkGLError();
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, gldata->indexArray.size() * sizeof(uint32_t), gldata->indexArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void surfelCloudRenderer::reloadColors(const pcl::PointCloud<rgbd::surfelPt>& cloud, const surfelColoringFunc& getSurfelColor)
{
	ASSERT_ALWAYS(cloud.points.size() * (gldata->trianglesPerSurfel + 1) == gldata->colorArray.size());
	for(unsigned int i = 0; i < cloud.points.size(); i++)
	{
		const size_t istart = i * (gldata->trianglesPerSurfel + 1);
		for(unsigned int j = 0; j < gldata->trianglesPerSurfel + 1; j++)
			gldata->colorArray[istart + j] = getSurfelColor(i, cloud.points[i]);
	}

	glBindBuffer(GL_ARRAY_BUFFER, gldata->cvboID);
	checkGLError();
	glBufferData(GL_ARRAY_BUFFER, gldata->colorArray.size() * sizeof(std::array<uint8_t, 4>), gldata->colorArray.data(), GL_STATIC_DRAW);
	checkGLError();
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void surfelCloudRenderer::renderAux() const
{
	glBindBuffer(GL_ARRAY_BUFFER, gldata->vvboID);
	glVertexPointer(3, GL_FLOAT, 0, 0/* offset into bound buffer */);
//	checkGLError();
	glBindBuffer(GL_ARRAY_BUFFER, gldata->cvboID);
	glColorPointer(4, GL_UNSIGNED_BYTE, 0, 0/* offset into bound buffer */);
//	checkGLError();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gldata->ivboID);
//	checkGLError();
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
//	checkGLError();
	glDrawElements(GL_TRIANGLES, gldata->indexArray.size(), GL_UNSIGNED_INT, 0/* offset into bound buffer */);
//	checkGLError();
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
//	checkGLError();
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
