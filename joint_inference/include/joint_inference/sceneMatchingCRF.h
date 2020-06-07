/*
 * sceneMatchingCRF: code to set up and run CRF inference for scene matching
 *
 * Evan Herbst
 * 11 / 4 / 10
 */

#ifndef EX_SCENE_MATCHING_CRF_H
#define EX_SCENE_MATCHING_CRF_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem/path.hpp>
#include "scene_matching/sceneMatchingCommon.h"
namespace fs = boost::filesystem;

/*
 * likeLikes, likeDislikes: scene -> seg 1 -> seg2
 * segCorrs: scene 1 -> scene 2 -> scene-1 seg -> scene-2 segs it can correspond with
 */
void twoSceneSceneMatchingCRFInference(const std::vector<boost::shared_ptr<sceneInfo>>& scenes, const unsigned int scene1index, const unsigned int scene2index,
	std::vector<std::unordered_map<unsigned int, std::unordered_set<unsigned int>>>& likeLikes,
	std::vector<std::unordered_map<unsigned int, std::unordered_set<unsigned int>>>& likeDislikes,
	std::vector<std::vector<std::unordered_map<unsigned int, std::unordered_set<unsigned int>>>>& segCorrs,
	const fs::path& outdir, const std::string& segFilebase);

#endif //header
