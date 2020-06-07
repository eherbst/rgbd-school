/*
 * diffingConnComps: find conn comps of high-p(m) pixels
 *
 * Evan Herbst
 * 1 / 15 / 14
 */

#include <boost/pending/disjoint_sets.hpp>
#include "rgbd_util/timer.h"
#include "rgbd_util/mathUtils.h"
#include "pcl_rgbd/pointTypes.h"
#include "pcl_rgbd/depth_to_cloud_lib.h"
#include "pcl_rgbd/cloudSearchTrees.h"
#include "active_obj_modeling/diffingConnComps.h"

/*
 * list conn comps of moved pixels (not including nonmoved pixels)
 *
 * probMovedThreshold: minimum p(m) to say that a pixel is likely moved
 *
 * distanceThreshold: for 3-d, max 3-d distance btwn pts to put into same component (not multiplied by depth-dependent factor); for 2-d, max delta-z btwn nbr pixels to put into same component (multiplied by depth-dependent factor; .008 is a good guess)
 *
 * return: component id (not zero-indexed) -> list of row-major pixels in comp, for all comps in img
 */
std::unordered_map<size_t, std::vector<size_t>> findDiffingConnComps(const rgbd::CameraParams& camParams, const cv::Mat_<float>& depth, const rgbd::eigen::VectorXd& movedProbs, const double probMovedThreshold, const diffingConnCompsAlgorithm alg,
	const float distanceThreshold)
{
	std::vector<bool> highProbMoved(movedProbs.size(), false);
	for(size_t i = 0; i < movedProbs.size(); i++)
		if(movedProbs[i] > probMovedThreshold)
			highProbMoved[i] = true;

	std::vector<size_t> rankMap(camParams.yRes * camParams.xRes), parentMap(camParams.yRes * camParams.xRes);
	boost::iterator_property_map<std::vector<size_t>::iterator, boost::typed_identity_property_map<size_t>> rankMapAdaptor(rankMap.begin()), parentMapAdaptor(parentMap.begin());
	boost::disjoint_sets<decltype(rankMapAdaptor), decltype(parentMapAdaptor)> sets(rankMapAdaptor, parentMapAdaptor);
	for(size_t i = 0, l = 0; i < camParams.yRes; i++)
		for(size_t j = 0; j < camParams.xRes; j++, l++)
			if(depth(i, j) > 0 && highProbMoved[l])
				sets.make_set(l);

	rgbd::timer t2;
	switch(alg)
	{
		case diffingConnCompsAlgorithm::FULL_3D: //conn comps in 3-d: gives a better segmentation than 2.5-d, but for pickupLunchboxWithBkgndMotion1, takes 2.7s/frame with r=.03 if initialRadius=.01
		{
			/*
			 * restrict the initial search to a smallish radius to keep time down; then merge the resulting components with another round of (less expensive) searches
			 */
			const float initialRadius = std::min(distanceThreshold, .01f);

			pcl::PointCloud<rgbd::pt> cloud;
			depth_to_cloud(depth, true/* include xy */, false, cloud, camParams);
			const boost::shared_ptr<kdtree2> kdtree = rgbd::createKDTree2(cloud);
			const kdtreeNbrhoodSpec nspec = kdtreeNbrhoodSpec::byRadius(initialRadius);
			for(size_t i = 0; i < cloud.points.size(); i++)
			{
				const size_t l = cloud.points[i].imgY * camParams.xRes + cloud.points[i].imgX;
				const std::vector<float> qpt = {cloud.points[i].x, cloud.points[i].y, cloud.points[i].z};
				const std::vector<kdtree2_result> nbrs = searchKDTree(*kdtree, nspec, qpt);
				for(const auto n : nbrs)
					if(n.idx > i) //process each pair only once
					{
						const size_t ln = cloud.points[n.idx].imgY * camParams.xRes + cloud.points[n.idx].imgX;
						if((movedProbs[l] > probMovedThreshold) == (movedProbs[ln] > probMovedThreshold)) //if both points have the same movedness status
							sets.union_set(l, ln);
					}
			}

			if(initialRadius < distanceThreshold) //if second level of merging is necessary
			{
				std::unordered_map<size_t, std::vector<size_t>> compsByRep; //representative -> list of pixels in comp
				for(size_t i = 0, l = 0; i < camParams.yRes; i++)
					for(size_t j = 0; j < camParams.xRes; j++, l++)
						if(depth(i, j) > 0 && highProbMoved[l])
							compsByRep[sets.find_set(l)].push_back(l);

				const float maxPixelDist = 12; //max distance in pixel space over which to merge comps; TODO ?
				const kdtreeNbrhoodSpec nspec = kdtreeNbrhoodSpec::byCount(1);
				std::vector<std::pair<size_t, size_t>> compsToMerge; //pairs of representatives
				for(const auto& c : compsByRep)
				{
					pcl::PointCloud<rgbd::pt> compCloud;
					compCloud.points.resize(c.second.size());
					copySelectedIndices(cloud.points, c.second, compCloud.points.begin());
					compCloud.width = compCloud.points.size();
					compCloud.height = 1;
					const boost::shared_ptr<kdtree2> compTree = rgbd::createKDTree2(compCloud);

					for(const auto& c2 : compsByRep)
						if(c2.second.size() < c.second.size()) //enumerate comps smaller than the one we indexed
						{
							float minSqrDist = FLT_MAX;
							for(size_t l : c2.second)
							{
								const std::vector<float> qpt = {cloud.points[l].x, cloud.points[l].y, cloud.points[l].z};
								const std::vector<kdtree2_result> nbrs = searchKDTree(*compTree, nspec, qpt);
								ASSERT_ALWAYS(nbrs.size() == 1);
								const float pixelDist = sqr((c.second[nbrs[0].idx] / camParams.xRes) - (l / camParams.xRes)) + sqr((c.second[nbrs[0].idx] % camParams.xRes) - (l % camParams.xRes)); //distance btwn these two points in pixel space
								if(nbrs[0].dis < minSqrDist && pixelDist < sqr(maxPixelDist)) minSqrDist = nbrs[0].dis;
							}
							if(minSqrDist < sqr(distanceThreshold)) compsToMerge.push_back(std::make_pair(c.first, c2.first));
						}
				}
				for(const auto& p : compsToMerge) sets.link(p.first, p.second);
			}
			break;
		}
		case diffingConnCompsAlgorithm::LARGE_NBRHOOD_2D: //conn comps in 2.5-d w/ a large nbrhood: a somewhat better segmentation than naive 2.5-d but about .1s/frame for hw=4, .06s/frame for hw=3
		{
			const int64_t nbrhoodHalfwidth = 4; //TODO ?
			for(int64_t i = 0, l = 0; i < camParams.yRes; i++)
				for(int64_t j = 0; j < camParams.xRes; j++, l++)
					if(depth(i, j) > 0 && highProbMoved[l])
						for(int64_t ii = i; ii <= std::min((int64_t)camParams.yRes - 1, i + nbrhoodHalfwidth); ii++)
							for(int64_t jj = j; jj <= std::min((int64_t)camParams.xRes - 1, j + nbrhoodHalfwidth); jj++)
							{
								const size_t ll = ii * camParams.xRes + jj;
								if(depth(ii, jj) > 0 && highProbMoved[ll])
									if(fabs(depth(ii, jj) - depth(i, j)) < distanceThreshold * sqrt(sqr(ii - i) + sqr(jj - j)) * primesensor::stereoErrorRatio(std::min(depth(i, j), depth(ii, jj))))
										sets.union_set((size_t)l, ll);
							}
			break;
		}
		case diffingConnCompsAlgorithm::SMALL_NBRHOOD_2D: //conn comps in 2.5-d: about .007s/frame
		{
			for(size_t i = 0, l = 0; i < camParams.yRes; i++)
				for(size_t j = 0; j < camParams.xRes; j++, l++)
					if(depth(i, j) > 0)
					{
						if(j < camParams.xRes - 1 && fabs(depth(i, j + 1) - depth(i, j)) < distanceThreshold * primesensor::stereoErrorRatio(depth(i, j)) && (movedProbs[l] > probMovedThreshold) == (movedProbs[l + 1] > probMovedThreshold)) sets.union_set(l, l + 1);
						if(i < camParams.yRes - 1 && fabs(depth(i + 1, j) - depth(i, j)) < distanceThreshold * primesensor::stereoErrorRatio(depth(i, j)) && (movedProbs[l] > probMovedThreshold) == (movedProbs[l + camParams.xRes] > probMovedThreshold)) sets.union_set(l, l + camParams.xRes);
					}
			break;
		}
	}
	t2.stop("run conn comps");

	std::unordered_map<size_t, std::vector<size_t>> compsByRep;
	for(size_t i = 0, l = 0; i < camParams.yRes; i++)
		for(size_t j = 0; j < camParams.xRes; j++, l++)
			if(depth(i, j) > 0 && highProbMoved[l])
				compsByRep[sets.find_set(l)].push_back(l);
	return compsByRep;
}
