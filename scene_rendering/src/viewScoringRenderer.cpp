/*
 * viewScoringRenderer: render views of a scene for scoring them as next best views
 *
 * Evan Herbst
 * 11 / 26 / 11
 */

//code for scoring in cuda runs but isn't tested; and not sure what the speedup is (that and accuracy are why I wrote it)
#define USE_MIPMAPS //for scoring; otherwise use cuda; TODO allow to disable at runtime if we won't be doing batch rendering?

#include <cassert>
#include <array>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <boost/format.hpp>
#include <boost/multi_array.hpp>
#include <boost/lexical_cast.hpp>
// in order to get function prototypes from gl.h and glext.h, define GL_GLEXT_PROTOTYPES first
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "rgbd_util/assert.h"
#include "rgbd_util/timer.h"
#include "opengl_util/renderingUtils.h"
#include "opengl_util/glInfo.h"
#include "opengl_util/openglContext.h"
#ifndef USE_MIPMAPS
#include "cuda_util/cudaUtils.h"
#include <cuda_gl_interop.h>
#endif
#include "scene_rendering/viewScoringRenderer.h"
using std::vector;
using std::cout;
using std::endl;
using rgbd::eigen::Vector3f;

#ifndef USE_MIPMAPS
/*
 * from viewScoringRenderer.cu
 */

/*
 * return scores (sum of green channel) for scenes in row-major order
 */
std::vector<float> scoreViewsInCUDA(const cudaArray* cudaColorBufferRGBA, const size_t bufWidth, const size_t bufHeight, const size_t numScenesX, const size_t numScenesY);
#endif

/**********************************************************************************/

/*
 * smallest power of 2 that's >= i
 * (the value)
 */
uint32_t roundUpPow2(const uint32_t i)
{
	ASSERT_ALWAYS(i > 0);
	uint32_t n = 1;
	while(n < i) n <<= 1;
	return n;
}

/*
 * largest power of 2 that's <= i
 * (the value)
 */
uint32_t roundDownPow2(const uint32_t i)
{
	const uint32_t n = roundUpPow2(i);
	return (n == i) ? n : n >> 1;
}

/*
 * what power of 2 is i, assuming it is exactly one of them
 * (the exponent)
 */
uint32_t whatPow2(uint32_t i)
{
	ASSERT_ALWAYS(i > 0);
	uint32_t e = 0;
	while(i > 1) {i >>= 1; e++;}
	return e;
}

/**********************************************************************************/

struct viewScoringRenderer::renderCallbacksData
{
	renderCallbacksData() : locked(0)
	{}

	std::vector<std::function<void (const rgbd::eigen::Affine3f& xform)>> renderSceneAux; //keep a list of renderers and allow to push and pop
	std::recursive_mutex renderFuncsMux;
	size_t locked; //how many levels of locking we're at right now (0 = unlocked)
};

struct viewScoringRenderer::openglData
{
	std::shared_ptr<openglContext> context; //not necessarily allocated by us

	/*
	 * opengl stuff
	 */
	GLuint fboId; // ID of FBO
	GLuint textureId; // ID of texture
	GLuint rboId; // ID of Renderbuffer object
	bool fboUsed; //if framebuffer objects aren't available we have a fallback mode

	uint32_t textureWidth, textureHeight; //we might need these to be powers of 2 for mipmapping to be fast
	uint32_t sceneWidth, sceneHeight; //used only if we're rendering many scenes on one texture
#ifdef USE_MIPMAPS
#else
	cudaGraphicsResource* cudaGLColorBuffer;
#endif
};

/*
 * if largeTexture is true, allocate a texture large enough to score many views at once; else allocate the size of camParams
 *
 * post: our opengl context is current
 */
viewScoringRenderer::viewScoringRenderer(const rgbd::CameraParams& c, const bool largeTexture)
: viewScoringRenderer(std::shared_ptr<openglContext>(), c, largeTexture)
{
}

viewScoringRenderer::viewScoringRenderer(const std::shared_ptr<openglContext>& ctx, const rgbd::CameraParams& c, const bool largeTexture)
: camParams(c), callbacksData(new renderCallbacksData), gldata(new openglData)
{
	init(ctx, largeTexture);
}

/*
 * shallow-copy the existing renderer's gl context and framebuffer objects
 */
viewScoringRenderer::viewScoringRenderer(const viewScoringRenderer& r, const rgbd::CameraParams& c)
: camParams(c)
{
	gldata = r.gldata;
}

viewScoringRenderer::~viewScoringRenderer()
{
#ifndef USE_MIPMAPS
	CUDA_CALL(cudaGraphicsUnregisterResource(gldata->cudaGLColorBuffer));
#endif

	glDeleteTextures(1, &gldata->textureId);
	checkGLError();

	if(gldata->fboUsed)
	{
		glDeleteFramebuffers(1, &gldata->fboId);
		glDeleteRenderbuffers(1, &gldata->rboId);
		checkGLError();
	}
}

/*
 * fill gldata
 *
 * create a gl context if ctx is empty
 */
void viewScoringRenderer::init(const std::shared_ptr<openglContext>& ctx, const bool largeTexture)
{
	bkgndCol = rgbd::eigen::Vector3f::Zero(); //zero is what we need for ID rendering

	if(ctx) gldata->context = ctx;
	else gldata->context.reset(new openglContext(camParams.xRes, camParams.yRes));
	/*
	 * make the context current
	 * (http://www.opengl.org/wiki/OpenGL_context: "in order for any OpenGL commands to work, a context must be current")
	 */
	gldata->context->acquire();

	// get OpenGL info
	glInfo glInfo;
	glInfo.getInfo();
//	glInfo.printSelf();

	/*
	 * init GL
	 */

	glShadeModel(GL_SMOOTH);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);

	/*
	 * initialize buffer objects
	 */

	if(largeTexture)
	{
		/*
		 * figure out how big a texture buffer we can use (we'll render a scene from many viewpoints into one texture buffer)
		 */
		//TODO use the more accurate "texture proxy" approach a la http://www.opengl.org/resources/faq/technical/texture.htm #21.130?
		const uint32_t sceneSize = 256; //width and height of each rendered scene -- TODO does it need to be a power of 2 for mipmapping to work?
		GLint maxTexSize;
		glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexSize);
		checkGLError();
		ASSERT_ALWAYS(maxTexSize >= sceneSize); //so we can render at least one scene per buffer
		/*
		 * when I get into really really big textures I get freezes more often than otherwise, so keep the textures
		 * below the max the card supports
		 *
		 * timing #s on uther:
		 * - with the quadro nvs 420
		 *   - tex size 2048, scene size 256: 97s to render 4200 views of 1.1M quads
		 *   - tex size 8192, scene size 256: 82s to render 4200 views of 1.1M quads
		 * - with the geforce gtx 580
		 *   - tex size 2048, scene size 256: 11.6s to render 6500 views of 1.6M quads
		 *   - tex size 4096, scene size 256: 11.0s to render 6500 views of 1.6M quads
		 *   - tex size 8192, scene size 256: 11.2s to render 6500 views of 1.6M quads
		 *   - tex size 4096, scene size 128: 11.4s to render 6500 views of 1.6M quads
		 */
		const uint32_t preferredTexSize = 2048;//4096; //the smaller I make these, the more I can allocate
		if(maxTexSize >= preferredTexSize)
		{
			gldata->textureWidth = preferredTexSize;
			gldata->textureHeight = preferredTexSize;
		}
		else
		{
			gldata->textureWidth = roundDownPow2(maxTexSize);
			gldata->textureHeight = roundDownPow2(maxTexSize);
		}
		ASSERT_ALWAYS(gldata->textureWidth >= camParams.xRes && gldata->textureHeight >= camParams.yRes); //we sometimes render only one scene but at camera resolution
		gldata->sceneWidth = sceneSize;
		gldata->sceneHeight = sceneSize;
		cout << "got tex size " << gldata->textureWidth << " x " << gldata->textureHeight << endl;
	}
	else //plan on rendering only one full-size scene at a time
	{
		gldata->textureWidth = roundUpPow2(camParams.xRes);
		gldata->textureHeight = roundUpPow2(camParams.yRes);
		GLint maxTexSize;
		glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexSize);
		checkGLError();
		ASSERT_ALWAYS(maxTexSize >= gldata->textureWidth && maxTexSize >= gldata->textureHeight);

		//just in case we do use this renderer for rendering many scenes later
		gldata->sceneWidth = gldata->textureWidth;
		gldata->sceneHeight = gldata->textureHeight;
	}

	/*
	 * create a texture object
	 */
	 glGenTextures(1, &gldata->textureId);
	 checkGLError();
	 glBindTexture(GL_TEXTURE_2D, gldata->textureId);
	 ASSERT_ALWAYS(glIsTexture(gldata->textureId));
	 checkGLError();
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	 glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, gldata->textureWidth, gldata->textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); //NULL means reserve memory but don't init it
	 checkGLError();
#ifdef USE_MIPMAPS
	 glGenerateMipmap(GL_TEXTURE_2D); //allocate mipmaps (20111123: doesn't improve speed, and will happen at the next glGenerateMipmap call anyway)
	 checkGLError();
#endif
	 glBindTexture(GL_TEXTURE_2D, 0);

	 if(glInfo.isExtensionSupported("GL_EXT_framebuffer_object"))
	 {
		 gldata->fboUsed = true;
		 cout << "Video card supports GL_EXT_framebuffer_object." << endl;
	 }
	 else
	 {
		 gldata->fboUsed = false;
		 cout << "Video card does NOT support GL_EXT_framebuffer_object." << endl;
	 }

	 if(gldata->fboUsed)
	{
		// create a renderbuffer object to store depth info
		// NOTE: A depth renderable image should be attached the FBO for depth test.
		// If we don't attach a depth renderable image to the FBO, then
		// the rendering output will be corrupted because of missing depth test.
		// If you also need stencil test for your rendering, then you must
		// attach additional image to the stencil attachment point, too.
		glGenRenderbuffers(1, &gldata->rboId);
		checkGLError();
		glBindRenderbuffer(GL_RENDERBUFFER, gldata->rboId);
		ASSERT_ALWAYS(glIsRenderbuffer(gldata->rboId));
		checkGLError();
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, gldata->textureWidth, gldata->textureHeight);
		checkGLError();
		glBindRenderbuffer(GL_RENDERBUFFER, 0);

		// create a framebuffer object, you need to delete them when program exits.
		glGenFramebuffers(1, &gldata->fboId);
		checkGLError();
		glBindFramebuffer(GL_FRAMEBUFFER, gldata->fboId);
		ASSERT_ALWAYS(glIsFramebuffer(gldata->fboId));
		checkGLError();

		// attach a texture to FBO color attachment point
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gldata->textureId, 0);
		checkGLError();

		// attach a renderbuffer to depth attachment point
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gldata->rboId);
		checkGLError();

		bool status = checkFramebufferStatus();
		if(!status) gldata->fboUsed = false;

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

#ifndef USE_MIPMAPS
	 CUDA_CALL(cudaGraphicsGLRegisterImage(&gldata->cudaGLColorBuffer, gldata->textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
#endif

	 gldata->context->release();
}

std::shared_ptr<openglContext> viewScoringRenderer::getContext()
{
	return gldata->context;
}

/*
 * throw if we aren't using an fbo
 */
GLuint viewScoringRenderer::getFramebufferObjectID() const
{
	if(gldata->fboUsed) return gldata->fboId;
	else throw std::runtime_error("not using fbo");
}
GLuint viewScoringRenderer::getDepthBufferObjectID() const
{
	if(gldata->fboUsed) return gldata->rboId;
	else throw std::runtime_error("not using fbo");
}
GLuint viewScoringRenderer::getColorBufferObjectID() const
{
	if(gldata->fboUsed) return gldata->textureId;
	else throw std::runtime_error("not using fbo");
}

/*
 * explicit synchronization to allow for uninterrupted access to render func settings, like
 *
 * acquire()
 * optionally setRenderFunc()
 * render()
 * optionally restoreRenderFunc()
 * release()
 */
void viewScoringRenderer::acquire()
{
	gldata->context->acquire(); //enforce a lock ordering on our context's lock and our own
	callbacksData->renderFuncsMux.lock();
	callbacksData->locked++;
}
void viewScoringRenderer::release()
{
	callbacksData->locked--;
	callbacksData->renderFuncsMux.unlock();
	gldata->context->release(); //enforce a lock ordering on our context's lock and our own
}

/*
 * must be called before rendering anything
 */
void viewScoringRenderer::setRenderFunc(const std::function<void (const rgbd::eigen::Affine3f& camPose)>& renderFunc)
{
	ASSERT_ALWAYS(callbacksData->locked > 0); //this won't actually enforce locking, so it's only here to catch some mistakes
	callbacksData->renderSceneAux.push_back(renderFunc);
}

/*
 * restore previously set render func
 */
void viewScoringRenderer::restoreRenderFunc()
{
	ASSERT_ALWAYS(callbacksData->locked > 0); //this won't actually enforce locking, so it's only here to catch some mistakes
	callbacksData->renderSceneAux.pop_back();
}

void viewScoringRenderer::setBkgndCol(const rgbd::eigen::Vector3f& rgb)
{
	bkgndCol = rgb;
}

/*
 * we don't always want to do this at the beginning of renderScene(), eg when we're rendering many scenes into one texture buffer
 */
void viewScoringRenderer::clearGLBuffers()
{
	if(gldata->fboUsed) glBindFramebuffer(GL_FRAMEBUFFER, gldata->fboId);

	glClearColor(bkgndCol[0], bkgndCol[1], bkgndCol[2], 0); //the color for free space
	glClearDepth(1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(gldata->fboUsed) glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

/*
 * clipping planes used for all rendering
 */
float viewScoringRenderer::zNear() const {return 1e-5;}
float viewScoringRenderer::zFar() const {return 1e2;}

/*
 * main rendering function (can be used to render to texture or screen)
 */
void viewScoringRenderer::renderScene(const rgbd::eigen::Affine3f& camPose, const uint16_t w, const uint16_t h, const uint16_t sceneXIndex, const uint16_t sceneYIndex)
{
	ASSERT_ALWAYS(callbacksData->locked > 0);

	const uint32_t viewportX0 = w * sceneXIndex, viewportY0 = h * sceneYIndex,
		viewportW = w, viewportH = h;
	glViewport(viewportX0, viewportY0, viewportW, viewportH);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//variable names as per http://www.songho.ca/opengl/gl_projectionmatrix.html
	const float n = zNear(), f = zFar(), r = camParams.xRes * .5 * n / camParams.focalLength, t = camParams.yRes * .5 * n / camParams.focalLength;
	std::array<GLfloat, 16> pmtx = //col-major
	{{
		n / r, 0, 0, 0,
		0, n / t, 0, 0,
		0, 0, -(f + n) / (f - n), -1,
		0, 0, -2 * f * n / (f - n), 0
	}};
	pmtx[5] *= -1; //correct for opengl having y increase as we go up rows
	glLoadMatrixf(pmtx.data());

	callbacksData->renderSceneAux.back()(camPose); //will set the modelview matrix
}

void viewScoringRenderer::renderToTexture(const std::function<void ()>& renderSceneFunc)
{
	/*
	 * start render
	 */
	if(gldata->fboUsed)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, gldata->fboId);
		checkGLError();
		glDrawBuffer(GL_COLOR_ATTACHMENT0);
		checkGLError();
	}
	else
	{
		glPushAttrib(GL_COLOR_BUFFER_BIT | GL_PIXEL_MODE_BIT); // for GL_DRAW_BUFFER and GL_READ_BUFFER
		glDrawBuffer(GL_FRONT); //probably use GL_BACK if doing double buffering
	}

	/*
	 * main rendering
	 */
	renderSceneFunc();

	/*
	 * end render
	 */
	if(gldata->fboUsed)
	{
#if 0
		//see what's in the fbo
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		checkGLError();
		cv::Mat img(gldata->textureHeight, gldata->textureWidth, CV_8UC3);
		glReadPixels(0, 0, gldata->textureWidth, gldata->textureHeight, GL_BGR, GL_UNSIGNED_BYTE, img.data);
		checkGLError();
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		checkGLError();
		cv::imwrite("render1.png", img);
		ASSERT_ALWAYS(false);
#endif

		// back to normal window-system-provided framebuffer
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		checkGLError();
	}
	else
	{
		// copy the framebuffer pixels to a texture
		glReadBuffer(GL_FRONT);
		glBindTexture(GL_TEXTURE_2D, gldata->textureId);
		glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, gldata->textureWidth, gldata->textureHeight);
		glBindTexture(GL_TEXTURE_2D, 0);

		glPopAttrib(); // GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT
		checkGLError();
	}
}

/*
 * render and copy to an image so the client can score the view
 * (see the code for what image formats we can write)
 *
 * pre: outImg has been allocated; its size matches our camera params
 */
void viewScoringRenderer::render(const rgbd::eigen::Affine3f& camPose, cv::Mat& outImg)
{
	gldata->context->acquire();
	clearGLBuffers();

	renderToTexture([this,&camPose](){renderScene(camPose, camParams.xRes, camParams.yRes, 0, 0);});

	/*
	 * copy to user buffer
	 */
	glFinish(); //TODO need?
	if(gldata->fboUsed)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, gldata->fboId);
		checkGLError();
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		checkGLError();
	}
	else
	{
		glReadBuffer(GL_FRONT);
		checkGLError();
	}
	ASSERT_ALWAYS(outImg.rows == camParams.yRes && outImg.cols == camParams.xRes);
	if(outImg.type() == cv::DataType<cv::Vec3b>::type) //probably meant for writing imgs to disk
		glReadPixels(0, 0, camParams.xRes, camParams.yRes, GL_BGR, GL_UNSIGNED_BYTE, outImg.data);
	else if(outImg.type() == cv::DataType<cv::Vec4b>::type) //used, eg, to render object IDs, so let's keep the byte order the same as in our inputs
		glReadPixels(0, 0, camParams.xRes, camParams.yRes, GL_RGBA, GL_UNSIGNED_BYTE, outImg.data);
	else ASSERT_ALWAYS(false);
	checkGLError();
	if(gldata->fboUsed)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	gldata->context->release();
}

/*
 * outDepth: should have been allocated (single-channel float); will be filled with values in meters
 */
void viewScoringRenderer::render(const rgbd::eigen::Affine3f& camPose, cv::Mat& outImg, cv::Mat& outDepth)
{
	gldata->context->acquire();
	clearGLBuffers();

	renderToTexture([this,&camPose](){renderScene(camPose, camParams.xRes, camParams.yRes, 0, 0);});

	/*
	 * copy to user buffers
	 */
	glFinish(); //TODO need?
	if(gldata->fboUsed)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, gldata->fboId);
		checkGLError();
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		checkGLError();
	}
	else
	{
		glReadBuffer(GL_FRONT);
		checkGLError();
	}
	//get color
	rgbd::timer t;
	ASSERT_ALWAYS(outImg.rows == camParams.yRes && outImg.cols == camParams.xRes);
	if(outImg.type() == cv::DataType<cv::Vec3b>::type) //probably meant for writing imgs to disk
		glReadPixels(0, 0, camParams.xRes, camParams.yRes, GL_BGR, GL_UNSIGNED_BYTE, outImg.data);
	else if(outImg.type() == cv::DataType<cv::Vec4b>::type) //used, eg, to render object IDs, so let's keep the byte order the same as in our inputs
		glReadPixels(0, 0, camParams.xRes, camParams.yRes, GL_RGBA, GL_UNSIGNED_BYTE, outImg.data);
	else ASSERT_ALWAYS(false);
	checkGLError();
	//get depth
	ASSERT_ALWAYS(outDepth.rows == camParams.yRes && outDepth.cols == camParams.xRes);
	ASSERT_ALWAYS(outDepth.type() == cv::DataType<float>::type);
	glReadPixels(0, 0, camParams.xRes, camParams.yRes, GL_DEPTH_COMPONENT, GL_FLOAT, outDepth.data);
	checkGLError();
	t.stop("readpixels");
	if(gldata->fboUsed)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	/*
	 * remap depths from [0, 1] to physical values
	 */
	const float zn = zNear(), zf = zFar();
	float* d = reinterpret_cast<float*>(outDepth.data);
	for(size_t i = 0; i < camParams.yRes; i++)
		for(size_t j = 0; j < camParams.xRes; j++, d++)
			*d = (zn * zf / (zn - zf)) / (*d - zf / (zf - zn));

	gldata->context->release();
}

void viewScoringRenderer::renderToPixelBufferObjects(const rgbd::eigen::Affine3f& camPose, GLuint colorPBOid, GLuint depthPBOid)
{
	gldata->context->acquire();
	clearGLBuffers();

	renderToTexture([this,&camPose](){renderScene(camPose, camParams.xRes, camParams.yRes, 0, 0);});

	/*
	 * copy to user buffers
	 */
	glFinish(); //TODO need?
	if(gldata->fboUsed)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, gldata->fboId);
		checkGLError();
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		checkGLError();
	}
	else
	{
		glReadBuffer(GL_FRONT);
		checkGLError();
	}
	//get color
	glBindBuffer(GL_PIXEL_PACK_BUFFER, colorPBOid);
	glReadPixels(0, 0, camParams.xRes, camParams.yRes, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	checkGLError();
	//get depth
	glBindBuffer(GL_PIXEL_PACK_BUFFER, depthPBOid);
	glReadPixels(0, 0, camParams.xRes, camParams.yRes, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	checkGLError();
	if(gldata->fboUsed)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	gldata->context->release();
}

/*
 * render to whatever buffer is active (don't, eg, copy to texture)
 */
void viewScoringRenderer::render(const rgbd::eigen::Affine3f& camPose)
{
	renderScene(camPose, camParams.xRes, camParams.yRes, 0, 0);
}

/*
 * render and score many views at once for efficiency
 *
 * return: scores for all views (score is average green-channel value of pixels in the view, with 0 used for background pixels)
 */
std::vector<float> viewScoringRenderer::renderAndScore(const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& camPoses)
{
	rgbd::timer t;
	vector<float> scores(camPoses.size());

	gldata->context->acquire();

	const uint32_t numScenesX = gldata->textureWidth / gldata->sceneWidth, numScenesY = gldata->textureHeight / gldata->sceneHeight;
	for(uint32_t i = 0, l = 0; l < camPoses.size(); i++)
	{
		clearGLBuffers();

		const uint32_t prevL = l;

		for(uint32_t j = 0; j < numScenesY; j++)
			for(uint32_t k = 0; k < numScenesX; k++, l++)
				if(l < camPoses.size())
					renderToTexture([this,k,j,l,&camPoses](){renderScene(camPoses[l], gldata->sceneWidth, gldata->sceneHeight, k, j);});

		/*
		 * score by summing green channel
		 */
#ifdef USE_MIPMAPS
		glBindTexture(GL_TEXTURE_2D, gldata->textureId);
	//	glHint(GL_GENERATE_MIPMAP_HINT, GL_FASTEST); //doesn't noticeably help
		const int mipmapLevel = std::max(whatPow2(gldata->textureWidth) - whatPow2(numScenesX), whatPow2(gldata->textureHeight) - whatPow2(numScenesY));
	//	cout << "level " << mipmapLevel << endl;
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, mipmapLevel);
		glEnable(GL_TEXTURE_2D); //for ATI cards -- see http://www.opengl.org/wiki/Common_Mistakes#Automatic_mipmap_generation
		glGenerateMipmap(GL_TEXTURE_2D);
		checkGLError();

		glFinish();

		boost::multi_array<std::array<uint8_t, 4>, 2> pixels(boost::extents[numScenesY][numScenesX]);
		glGetTexImage(GL_TEXTURE_2D, mipmapLevel, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
		checkGLError();

		glBindTexture(GL_TEXTURE_2D, 0);

#if 1 //not debugging
		for(uint32_t j = 0, l2 = prevL; j < numScenesY; j++)
			for(uint32_t k = 0; k < numScenesX; k++, l2++)
				if(l2 < camPoses.size())
					scores[l2] = (float)pixels[j][k][1] / 255;
#else //debugging
		cout << "buffer " << i << " scores: ";
		for(uint32_t j = 0, l2 = prevL; j < numScenesY; j++)
			for(uint32_t k = 0; k < numScenesX; k++, l2++)
				if(l2 < camPoses.size())
				{
					scores[l2] = (float)pixels[j][k][1] / 255;
					cout << scores[l2] << ' ';
				}
		cout << endl;
#endif
#else
		/*
		 * have to map at each use because opengl can move buffers around in memory
		 */
		CUDA_CALL(cudaGraphicsMapResources(1, &gldata->cudaGLColorBuffer, 0));
		CUDA_CALL(cudaGetLastError());
		cudaArray* cudaColorArray;
		CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&cudaColorArray, gldata->cudaGLColorBuffer, 0, 0));

		const std::vector<float> someScores = std::move(scoreViewsInCUDA(cudaColorArray, gldata->textureWidth, gldata->textureHeight, gldata->sceneWidth, gldata->sceneHeight));
		std::copy_n(someScores.begin(), std::min(size_t(numScenesX * numScenesY), camPoses.size() - prevL), scores.begin() + prevL);

		CUDA_CALL(cudaGraphicsUnmapResources(1, &gldata->cudaGLColorBuffer, 0));
#endif

#if 0
	{
		//see what's in the fbo
		glFinish();
		glBindFramebuffer(GL_FRAMEBUFFER, gldata->fboId);
		checkGLError();
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		checkGLError();
		cv::Mat img(gldata->textureHeight, gldata->textureWidth, CV_8UC3);
		glReadPixels(0, 0, gldata->textureWidth, gldata->textureHeight, GL_BGR, GL_UNSIGNED_BYTE, img.data);
		checkGLError();
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		checkGLError();
		cv::imwrite("fulltex" + boost::lexical_cast<std::string>(i) + ".png", img);
	}
#endif

#if 0
		for(uint32_t j = 0, l2 = prevL; j < numScenesY; j++)
			for(uint32_t k = 0; k < numScenesX; k++, l2++)
				if(l2 < camPoses.size())
				{
					cv::Mat img(480, 640, CV_8UC3);
					render(camPoses[l2], img);
					cv::imwrite("render1_" + boost::lexical_cast<std::string>(i) + "_" + boost::lexical_cast<std::string>(l2) + "_" + boost::lexical_cast<std::string>(scores[l2]) + ".png", img);
				}
#endif
	}

	t.stop((boost::format("render %1%") % camPoses.size()).str());
	gldata->context->release();
	return scores;
}
