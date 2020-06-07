/*
 * pointCloudRenderer: render a point cloud from many camera poses
 *
 * Evan Herbst
 * 2 / 20 / 12
 */

#include <cassert>
#include <iostream>
// in order to get function prototypes from gl.h and glext.h, define GL_GLEXT_PROTOTYPES first
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <opencv2/core/core.hpp>
#include "rgbd_util/timer.h"
#include "opengl_util/renderingUtils.h"
#include "scene_rendering/pointCloudRenderer.h"
using std::vector;
using std::cout;
using std::endl;
using rgbd::eigen::Vector3f;

/*
 * use point's color field
 */
std::array<uint8_t, 4> getRenderingColorFromPointColor(const uint32_t index, const rgbd::pt& pt)
{
	const boost::array<unsigned char, 3> rgb = rgbd::unpackRGB<unsigned char>(pt.rgb);
	return std::array<uint8_t, 4>{{rgb[0], rgb[1], rgb[2], 255}};
}

/*
 * use pt index
 */
std::array<uint8_t, 4> getPointColorFromID(const uint32_t index, const rgbd::pt& pt)
{
	std::array<uint8_t, 4> col;
	uint32_t* c = reinterpret_cast<uint32_t*>(col.data());
	*c = index + 1;
	return col;
}

/*
 * what we need in addition to what we get from viewScoringRenderer
 */
struct pointCloudRenderer::openglData
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

	uint32_t pointSize;
};

/*
 * pre: cloud has normals set
 */
pointCloudRenderer::pointCloudRenderer(const pcl::PointCloud<rgbd::pt>& cloud, const uint32_t ptSize, const rgbd::CameraParams& c)
: gldata(new openglData)
{
	gldata->pointSize = ptSize;
	init(cloud, getRenderingColorFromPointColor);
}
pointCloudRenderer::pointCloudRenderer(const pcl::PointCloud<rgbd::pt>& cloud, const pointColoringFunc& getPointColor, const uint32_t ptSize, const rgbd::CameraParams& c)
: gldata(new openglData)
{
	gldata->pointSize = ptSize;
	init(cloud, getPointColor);
}

pointCloudRenderer::~pointCloudRenderer()
{
	glDeleteBuffers(1, &gldata->vvboID);
	glDeleteBuffers(1, &gldata->cvboID);
	glDeleteBuffers(1, &gldata->ivboID);
	checkGLError();
}

void pointCloudRenderer::init(const pcl::PointCloud<rgbd::pt>& cloud, const pointColoringFunc& getPointColor)
{
	/*
	 * init vertex buffer objects (like server-side vertex arrays)
	 */
	checkGLError();
	glGenBuffers(1, &gldata->vvboID);
	checkGLError();
	glGenBuffers(1, &gldata->cvboID);
	checkGLError();
	glGenBuffers(1, &gldata->ivboID);
	checkGLError();

	gldata->vertexArray.resize(cloud.points.size());
	gldata->colorArray.resize(gldata->vertexArray.size());
	for(unsigned int i = 0; i < cloud.points.size(); i++)
	{
		gldata->vertexArray[i] = rgbd::ptX2eigen<rgbd::eigen::Vector3f>(cloud.points[i]);
		gldata->colorArray[i] = getPointColor(i, cloud.points[i]);
	}

	gldata->indexArray.resize(cloud.points.size());
	for(unsigned int i = 0; i < cloud.points.size(); i++)
		gldata->indexArray[i] = i;

	glBindBuffer(GL_ARRAY_BUFFER, gldata->vvboID);
	ASSERT_ALWAYS(glIsBuffer(gldata->vvboID));
	checkGLError();
	glBufferData(GL_ARRAY_BUFFER, gldata->vertexArray.size() * sizeof(rgbd::eigen::Vector3f), gldata->vertexArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ARRAY_BUFFER, gldata->cvboID);
	ASSERT_ALWAYS(glIsBuffer(gldata->cvboID));
	checkGLError();
	glBufferData(GL_ARRAY_BUFFER, gldata->colorArray.size() * sizeof(std::array<uint8_t, 4>), gldata->colorArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gldata->ivboID);
	ASSERT_ALWAYS(glIsBuffer(gldata->ivboID));
	checkGLError();
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, gldata->indexArray.size() * sizeof(uint32_t), gldata->indexArray.data(), GL_STATIC_DRAW);
	checkGLError();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void pointCloudRenderer::reloadColors(const pcl::PointCloud<rgbd::pt>& cloud, const pointColoringFunc& getPointColor)
{
	ASSERT_ALWAYS(cloud.points.size() == gldata->colorArray.size());
	for(unsigned int i = 0; i < cloud.points.size(); i++)
	{
		gldata->colorArray[i] = getPointColor(i, cloud.points[i]);
	}

	glBindBuffer(GL_ARRAY_BUFFER, gldata->cvboID);
	checkGLError();
	glBufferData(GL_ARRAY_BUFFER, gldata->colorArray.size() * sizeof(std::array<uint8_t, 4>), gldata->colorArray.data(), GL_STATIC_DRAW);
	checkGLError();
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void pointCloudRenderer::renderAux() const
{
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
	glPointSize(gldata->pointSize);
	glDrawElements(GL_POINTS, gldata->indexArray.size(), GL_UNSIGNED_INT, 0/* offset into bound buffer */);
	checkGLError();
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	checkGLError();
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}
