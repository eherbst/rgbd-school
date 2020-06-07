/*
 * diffingConnComps: find conn comps of high-p(m) pixels
 *
 * Evan Herbst
 * 1 / 15 / 14
 */

#ifndef EX_DIFFING_CONN_COMPS_H
#define EX_DIFFING_CONN_COMPS_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include "rgbd_util/eigen/Core"
#include "rgbd_util/CameraParams.h"

enum class diffingConnCompsAlgorithm {SMALL_NBRHOOD_2D, LARGE_NBRHOOD_2D, FULL_3D}; //3-d is slowest but most accurate
/*
 * list conn comps of moved pixels (not including nonmoved pixels)
 *
 * probMovedThreshold: minimum p(m) to say that a pixel is likely moved
 *
 * distanceThreshold: for 3-d, max 3-d distance btwn pts to put into same component (not multiplied by depth-dependent factor); for 2-d, max delta-z btwn nbr pixels to put into same component (multiplied by depth-dependent factor; .008 is a good guess)
 *
 * return: component id (not zero-indexed) -> list of row-major pixels in comp, for all comps in img
 */
std::unordered_map<size_t, std::vector<size_t>> findDiffingConnComps(const rgbd::CameraParams& camParams, const cv::Mat_<float>& depth, const rgbd::eigen::VectorXd& movedProbs, const double probMovedThreshold,
	const diffingConnCompsAlgorithm alg = diffingConnCompsAlgorithm::LARGE_NBRHOOD_2D, const float distanceThreshold = .008);

#endif //header
