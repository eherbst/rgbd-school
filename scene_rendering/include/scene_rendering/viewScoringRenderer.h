/*
 * viewScoringRenderer: render views of a scene for scoring them as next best views
 *
 * Evan Herbst
 * 11 / 26 / 11
 */

#ifndef EX_VIEW_SCORING_RENDERER_H
#define EX_VIEW_SCORING_RENDERER_H

#include <functional>
#include <memory>
#include <GL/gl.h>
#include <opencv2/core/core.hpp>
#include "rgbd_util/CameraParams.h"
#include "rgbd_util/eigen/Geometry"
#include "rgbd_util/eigen/StdVector"
#include "opengl_util/openglContext.h"

/*
 * store opengl buffers for rendering into, and provide a framework for rendering a scene as well as scoring many views of a scene based on user-defined scores
 *
 * not thread-safe because opengl calls are global
 */
class viewScoringRenderer
{
	public:

		/*
		 * if largeTexture is true, allocate a texture large enough to score many views at once; else allocate the size of camParams
		 *
		 * post: our opengl context is current
		 */
		viewScoringRenderer(const rgbd::CameraParams& c, const bool largeTexture = true);
		viewScoringRenderer(const std::shared_ptr<openglContext>& ctx, const rgbd::CameraParams& c, const bool largeTexture = true);
		/*
		 * shallow-copy the existing renderer's gl context and framebuffer objects
		 */
		viewScoringRenderer(const viewScoringRenderer& r, const rgbd::CameraParams& c);
		virtual ~viewScoringRenderer();

		std::shared_ptr<openglContext> getContext();

		/*
		 * explicit synchronization to allow for uninterrupted access to render func settings, like
		 *
		 * acquire()
		 * optionally setRenderFunc()
		 * render()
		 * optionally restoreRenderFunc()
		 * release()
		 */
		void acquire();
		void release();

		/*
		 * must be called before rendering anything
		 */
		void setRenderFunc(const std::function<void (const rgbd::eigen::Affine3f& camPose)>& renderFunc);
		/*
		 * restore previously set render func
		 */
		void restoreRenderFunc();

		void setBkgndCol(const rgbd::eigen::Vector3f& rgb);

		/*
		 * render and copy to an image so the client can score the view
		 * (see the code for what image formats we can write)
		 *
		 * pre: outImg has been allocated; its size matches our camera params
		 */
		void render(const rgbd::eigen::Affine3f& camPose, cv::Mat& outImg);
		/*
		 * outDepth: should have been allocated (single-channel float); will be filled with values in meters
		 */
		void render(const rgbd::eigen::Affine3f& camPose, cv::Mat& outImg, cv::Mat& outDepth);
		/*
		 * render to whatever buffer is active (don't, eg, copy to texture)
		 */
		void render(const rgbd::eigen::Affine3f& camPose);
		/*
		 * render and copy into PBOs, which can be on the gpu
		 */
		void renderToPixelBufferObjects(const rgbd::eigen::Affine3f& camPose, GLuint colorPBOid, GLuint depthPBOid);

		/*
		 * throw if we aren't using an fbo
		 */
		GLuint getFramebufferObjectID() const;
		GLuint getDepthBufferObjectID() const;
		GLuint getColorBufferObjectID() const;

		/*
		 * render and score many views at once for efficiency
		 *
		 * return: scores for all views (score is average green-channel value of pixels in the view, with 0 used for background pixels)
		 */
		std::vector<float> renderAndScore(const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& camPoses);

		/*
		 * clipping planes used for all rendering
		 */
		float zNear() const;
		float zFar() const;

	protected:

		/*
		 * fill gldata
		 *
		 * create a gl context if ctx is empty
		 */
		void init(const std::shared_ptr<openglContext>& ctx, const bool largeTexture);

		void renderToTexture(const std::function<void ()>& renderSceneFunc);
		/*
		 * main rendering function (can be used to render to texture or screen)
		 */
		void renderScene(const rgbd::eigen::Affine3f& camPose, const uint16_t w, const uint16_t h, const uint16_t sceneXIndex, const uint16_t sceneYIndex);

		void clearGLBuffers();

		rgbd::CameraParams camParams;

		rgbd::eigen::Vector3f bkgndCol;

		/*
		 * auxiliary to renderScene(): set up the modelview matrix and draw the scene
		 */
		struct renderCallbacksData;
		std::shared_ptr<renderCallbacksData> callbacksData;

		/*
		 * PIMPL, but also make dealing with glx header issues easier by keeping them in our .cpp
		 */
		struct openglData;
		std::shared_ptr<openglData> gldata;
};

#endif //header
