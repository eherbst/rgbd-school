/*
 * sceneRenderingCache: vertex buffer objects and such for rendering some representation of a scene
 *
 * Evan Herbst
 * 2 / 20 / 12
 */

#include <GL/gl.h>
#include <GL/glu.h>
#include "scene_rendering/sceneRenderingCache.h"
using rgbd::eigen::Vector3f;

void sceneRenderingCache::render(const rgbd::eigen::Affine3f& camPose) const
{
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
#if 1 //use camPose
	const Vector3f eye = camPose * Vector3f(0, 0, 0),
		viewDir = camPose.linear() * Vector3f(0, 0, 1),
		up = camPose.linear() * Vector3f(0, -1, 0);
//	cout << "e " << eye.transpose() << " ; l " << (eye + viewDir).transpose() << " ; u " << up.transpose() << endl;
	gluLookAt(eye.x(), eye.y(), eye.z(),  eye.x() + viewDir.x(), eye.y() + viewDir.y(), eye.z() + viewDir.z(),  up.x(), up.y(), up.z());
#else //for testing
	gluLookAt(0, 0, 0,  0, 0, 1,  0, -1, 0); // eye(x,y,z), focal(x,y,z), up(x,y,z)
#endif

	renderAux();

	glPopMatrix();
}
