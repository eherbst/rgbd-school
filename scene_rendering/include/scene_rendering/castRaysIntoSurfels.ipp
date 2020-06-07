/*
 * castRaysIntoSurfels: associate locations in an image plane with surfels in a cloud
 *
 * Evan Herbst
 * 12 / 13 / 10
 */

#include <cassert>
#include <cfloat> //DBL_MAX
#include <limits>
#include <boost/range/concepts.hpp>
#include <boost/range.hpp>
#include <boost/concept/assert.hpp>

/*
 * project a set of points into a camera's (u, v, z) frame
 *
 * camXform: cam coords -> coords of surfel cloud
 *
 * return: for each surfel, image (u, v, z) in (pixels, pixels, m) in the camera's frame
 * (may not be in image bounds; check if you care)
 */
template <typename PointT>
std::tuple<std::vector<rgbd::eigen::Vector2i>, std::vector<float>> projectAllCloudPointsIntoCamera(const pcl::PointCloud<PointT>& cloud, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose)
{
	std::tuple<std::vector<rgbd::eigen::Vector2i>, std::vector<float>> result;
	std::vector<rgbd::eigen::Vector2i>& coords2d = std::get<0>(result);
	std::vector<float>& depths = std::get<1>(result);
	coords2d.resize(cloud.points.size());
	depths.resize(cloud.points.size());

	const rgbd::eigen::Affine3f camPoseInv(camPose.inverse());

	/*
	 * project each surfel's center to a single pixel
	 *
	 * TODO deal with the fact that surfels can be multiple pixels wide?
	 */
	for(unsigned int i = 0; i < cloud.points.size(); i++)
	{
		const rgbd::eigen::Vector4f surfelPos = rgbd::ptX2eigen<rgbd::eigen::Vector4f>(cloud.points[i]);
		const rgbd::eigen::Vector4f surfelPosInCamFrame = camPoseInv * surfelPos;
		const rgbd::eigen::Vector2f pixelPos(camParams.focalLength * surfelPosInCamFrame.x() / surfelPosInCamFrame.z() + camParams.centerX,
										camParams.focalLength * surfelPosInCamFrame.y() / surfelPosInCamFrame.z() + camParams.centerY);
		const int x = rint(pixelPos.x()), y = rint(pixelPos.y());
		coords2d[i] = rgbd::eigen::Vector2i(x, y);
		depths[i] = surfelPosInCamFrame.z();
	}

	return result;
}

/*
 * project a set of 3-d points into a camera's frame
 *
 * camXform: cam coords -> coords of the cloud the pts3d came from
 *
 * return: for each pixel, id of closest point plus one and its distance (z in (u, v, z) ), or, if no such, id 0 and an undefined depth
 *
 * ForwardRangeT's value type should be convertible to Vector3f
 *
 * pre: ptIDs, ptDepths are allocated
 */
template <typename ForwardRangeT>
void projectPointsIntoCamera(const ForwardRangeT& pts3d, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose, boost::multi_array<uint32_t, 2>& ptIDs, boost::multi_array<float, 2>& ptDepths)
{
	BOOST_CONCEPT_ASSERT((boost::ForwardRangeConcept<ForwardRangeT>));
	ASSERT_ALWAYS(ptIDs.shape()[0] == camParams.yRes && ptIDs.shape()[1] == camParams.xRes);
	ASSERT_ALWAYS(ptDepths.shape()[0] == camParams.yRes && ptDepths.shape()[1] == camParams.xRes);
	std::fill(ptIDs.data(), ptIDs.data() + ptIDs.num_elements(), 0);
	std::fill(ptDepths.data(), ptDepths.data() + ptDepths.num_elements(), std::numeric_limits<float>::max());

	const rgbd::eigen::Affine3f camPoseInv(camPose.inverse());

	size_t i = 0;
	for(auto j = boost::begin(pts3d); j != boost::end(pts3d); j++, i++)
	{
		const rgbd::eigen::Vector4f ptPos((*j)[0], (*j)[1], (*j)[2], 1);
		const rgbd::eigen::Vector4f ptPosInCamFrame = camPoseInv * ptPos;
		const rgbd::eigen::Vector2f pixelPos(camParams.focalLength * ptPosInCamFrame.x() / ptPosInCamFrame.z() + camParams.centerX,
															camParams.focalLength * ptPosInCamFrame.y() / ptPosInCamFrame.z() + camParams.centerY);
		const int x = rint(pixelPos.x()), y = rint(pixelPos.y());
		if(x >= 0 && x < (int)camParams.xRes && y >= 0 && y < (int)camParams.yRes)
		{
			const double depth = ptPosInCamFrame.z();
			if(depth > 0 && depth < ptDepths[y][x])
			{
				ptDepths[y][x] = depth;
				ptIDs[y][x] = i + 1;
			}
		}
	}
}

/*
 * project a set of 3-d points into a camera's frame
 *
 * camXform: cam coords -> coords of the cloud the pts3d came from
 *
 * return: for each pt, image (u, v, z) in (pixels, pixels, m) in the camera's frame
 * (may not be in image bounds; check if you care)
 *
 * ForwardRangeT's value type should be convertible to Vector3f
 */
template <typename ForwardRangeT>
std::tuple<std::vector<rgbd::eigen::Vector2i>, std::vector<float>> projectAllPointsIntoCamera(const ForwardRangeT& pts3d, const rgbd::CameraParams& camParams, const rgbd::eigen::Affine3f& camPose)
{
	BOOST_CONCEPT_ASSERT((boost::ForwardRangeConcept<ForwardRangeT>));

	std::tuple<std::vector<rgbd::eigen::Vector2i>, std::vector<float>> result;
	std::vector<rgbd::eigen::Vector2i>& coords2d = std::get<0>(result);
	std::vector<float>& depths = std::get<1>(result);
	coords2d.resize(boost::size(pts3d));
	depths.resize(boost::size(pts3d));

	const rgbd::eigen::Affine3f camPoseInv(camPose.inverse());

	size_t i = 0;
	for(auto j = boost::begin(pts3d); j != boost::end(pts3d); j++, i++)
	{
		const rgbd::eigen::Vector4f ptPosInCamFrame = camPoseInv * rgbd::eigen::Vector4f((*j)[0], (*j)[1], (*j)[2], 1);
		const int x = rint(camParams.focalLength * ptPosInCamFrame.x() / ptPosInCamFrame.z() + camParams.centerX),
			y = rint(camParams.focalLength * ptPosInCamFrame.y() / ptPosInCamFrame.z() + camParams.centerY);
		coords2d[i] = rgbd::eigen::Vector2i(x, y);
		depths[i] = ptPosInCamFrame.z();
	}

	return result;
}
