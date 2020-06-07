/*
 * for the icra14 camera-ready video: highlight an obj model in the corresponding img of the full scene
 */

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "rgbd_util/assert.h"
namespace fs = boost::filesystem;

/*
 * arguments: full-scene rendering img, obj model rendering img, outpath
 */
int main(int argc, char* argv[])
{
	ASSERT_ALWAYS(argc == 4);
	const fs::path imgToEditPath = argv[1], imgForMaskPath = argv[2], outimgPath = argv[3];

	cv::Mat_<cv::Vec3b> imgToEdit = cv::imread(imgToEditPath.string()), imgForMask = cv::imread(imgForMaskPath.string());

	//make mask
	cv::Mat_<uint8_t> maskImg(imgForMask.size());
	for(size_t i = 0; i < imgForMask.rows; i++)
		for(size_t j = 0; j < imgForMask.cols; j++)
			if(imgForMask(i, j) != cv::Vec3b(255, 229, 204)) maskImg(i, j) = 0; //if not bkgnd
			else maskImg(i, j) = 255;

	//distance transform
	cv::Mat distsToSeg;
	cv::distanceTransform(maskImg, distsToSeg, CV_DIST_L2, CV_DIST_MASK_PRECISE);

	//edit img
	for(size_t i = 0; i < imgForMask.rows; i++)
		for(size_t j = 0; j < imgForMask.cols; j++)
			if(distsToSeg.at<float>(i, j) > 0 && distsToSeg.at<float>(i, j) < 2.5)
				imgToEdit(i, j) = cv::Vec3b(255, 160, 255);

	cv::imwrite(outimgPath.string(), imgToEdit);
	return 0;
}
