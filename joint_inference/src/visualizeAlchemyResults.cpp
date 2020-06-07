/*
 * visualizeAlchemyResults
 *
 * Evan Herbst
 * 8 / 27 / 10
 */

#include <cassert>
#include <cstdio> //getchar()
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <utility>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/format.hpp>
#include <boost/unordered_map.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/regex.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <pcl/point_types.h>
#include "rgbd_util/timer.h"
#include "rgbd_util/mathUtils.h"
#include "rgbd_util/ioUtils.h"
#include "rgbd_util/ros_utility.h"
#include "rgbd_util/serialization/multi_array.h"
#include "rgbd_util/serialization/eigen.h"
#include "xforms/xforms.h"
#include "evh_util/rosUtils.h" //hash<ros::Time>
#include "evh_util/fsUtils.h"
#include "evh_util/strongType.h"
#include "evh_util/serialization/unordered_map.h"
#include "evh_util/serialization/unordered_set.h"
#include "evh_util/visualizationUtils.h"
#include "probabilistic_surfels/readSurfelCloud.h"
#include "evh_util/pointsNLines.h"
#include "pcl_rgbd/cloudTofroPLY.h"
#include "scene_matching/sceneMatchingIO.h"
using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::set;
using std::string;
using std::ifstream;
using std::ofstream;
using std::ostringstream;
using std::cout;
using std::cerr;
using std::endl;
using std::pair;
using boost::lexical_cast;
namespace fs = boost::filesystem;
namespace ar = boost::archive;
using rgbd::eigen::Vector2i;
using rgbd::eigen::Vector3f;
using rgbd::eigen::Matrix4f;
using rgbd::eigen::AngleAxisf;
using rgbd::eigen::Translation3f;

/*
 * arguments: mln outdir, # scenes, dataset dir, scene names, scene #s to look for
 */
int main(int argc, char* argv[])
{
	ros::Time::init();
	ASSERT_ALWAYS(argc >= 2);

	/********************************************************************************************************
	 * read datasets
	 */

	unsigned int _ = 1;
	const fs::path mlndir(argv[_++]), smoutdir = mlndir.parent_path();
	const unsigned int numScenes = lexical_cast<unsigned int>(argv[_++]);
	ASSERT_ALWAYS(numScenes == 2); //or update this file
	const fs::path datasetDir = fs::path(argv[_++]);

	vector<sceneInfo> scenes(numScenes);
	for(unsigned int i = 0; i < numScenes; i++)
	{
		const string sceneName = argv[_++];
		sceneInitOptions opts;
		opts.computePrincipalCurvatures = false;
		opts.readSpinImgs = false;
		scenes[i] = std::move(initializeScene(datasetDir, sceneName, opts));
	}

	const vector<unsigned int> sceneIndices = {lexical_cast<unsigned int>(argv[_++]), lexical_cast<unsigned int>(argv[_++])};
	unordered_map<unsigned int, unsigned int> indexMap;
	for(unsigned int i = 0; i < sceneIndices.size(); i++) indexMap[sceneIndices[i]] = i;

	/*
	 * visualize foreground surfel graph
	 */
	vector<unordered_map<unsigned int, unsigned int>> surfel2undersegID(numScenes);
	for(unsigned int i = 0; i < numScenes; i++)
	{
		ifstream infile((smoutdir / ("fgSurfelgraph-" + scenes[i].sceneName + ".pnl")).string(), ifstream::binary);
		ASSERT_ALWAYS(infile);
		unsigned char pltype;
		unsigned int numSegments = 1; //if no pts have segs given, we'll draw them all the same col
		while(readBinary(infile, pltype))
		{
			switch(pltype)
			{
				case pnl::type::POINT:
				{
					unsigned int id;
					Vector3f f;
					pnl::readPoint(infile, id, f);
					surfel2undersegID[i][id] = 0; //we won't use underseg ids
					break;
				}
				case pnl::type::LINE:
				{
					unsigned int i1, i2;
					pnl::readLine(infile, i1, i2);
					break;
				}
				case pnl::type::POINT_WITH_SEG:
				{
					unsigned int id;
					Vector3f f;
					unsigned int segID;
					pnl::readPointWithSegID(infile, id, f, segID);
					break;
				}
				case pnl::type::WEIGHTED_LINE:
				{
					unsigned int i1, i2;
					float w;
					pnl::readWeightedLine(infile, i1, i2, w);
					break;
				}
				default: ASSERT_ALWAYS(false);
			}
		}
	}

	/********************************************************************************************************
	 * merge surfels/feats for each scene into some sort of segments of which we have a small enough number to run an mln over them
	 */

	rgbd::timer t;
	vector<unordered_map<unsigned int, unsigned int>> surfel2componentIndex(numScenes); //scene -> surfel -> component # for some subset of foreground surfels
	vector<unsigned int> numFGComponents(numScenes);
	boost::format surf2compFilepathFmt((smoutdir / "segsForXforms-%1%.out").string()); //args: scene name
	for(unsigned int i = 0; i < numScenes; i++)
	{
		const fs::path surf2compFilepath((surf2compFilepathFmt % scenes[i].sceneName).str());
		if(fs::exists(surf2compFilepath))
		{
			ifstream infile(surf2compFilepath.string().c_str(), ifstream::binary);
			ASSERT_ALWAYS(infile);
			ar::binary_iarchive reader(infile);
			reader >> surfel2componentIndex[i];

			numFGComponents[i] = 0;
			for(auto j = surfel2componentIndex[i].begin(); j != surfel2componentIndex[i].end(); j++)
				if((*j).second >= numFGComponents[i])
					numFGComponents[i] = (*j).second + 1;
		}
		else ASSERT_ALWAYS(false);
	}
	t.stop("get fg segs");
	for(unsigned int i = 0; i < numScenes; i++) cout << "scene " << i << ": " << numFGComponents[i] << " fg segs" << endl;
	/*
	 * consistency checks
	 */
	for(unsigned int i = 0; i < numScenes; i++)
	{
		unsigned int maxSeg = 0;
		for(auto j = surfel2componentIndex[i].begin(); j != surfel2componentIndex[i].end(); j++)
			if((*j).second > maxSeg)
				maxSeg = (*j).second;
		ASSERT_ALWAYS(maxSeg + 1 == numFGComponents[i]);
	}

	/*
	 * get segment spatial stats
	 */
	vector<vector<Vector3f>> segmentCentroids(numScenes); //scene -> seg -> seg centroid in scene's coords
	for(unsigned int i = 0; i < numScenes; i++)
	{
		segmentCentroids[i].resize(numFGComponents[i], Vector3f::Zero());
		vector<unsigned int> componentCounts(numFGComponents[i], 0);
		for(auto j = surfel2componentIndex[i].begin(); j != surfel2componentIndex[i].end(); j++)
		{
			segmentCentroids[i][(*j).second] += rgbd::ptX2eigen<Vector3f>(scenes[i].surfelCloudPtr->points[(*j).first]);
			componentCounts[(*j).second]++;
		}
		for(unsigned int j = 0; j < segmentCentroids[i].size(); j++) segmentCentroids[i][j] /= componentCounts[j];
	}

	t.restart();

	const bool inferenceIsMAP = true;

	/*
	read results file and interpret probabilities
	*/
	typedef boost::associative_property_map<boost::unordered_map<pair<unsigned int, unsigned int>, unsigned int> > rankMapT;
	typedef boost::associative_property_map<boost::unordered_map<pair<unsigned int, unsigned int>, pair<unsigned int, unsigned int>> > parentMapT;
	boost::unordered_map<pair<unsigned int, unsigned int>, unsigned int> rankMap;
	boost::unordered_map<pair<unsigned int, unsigned int>, pair<unsigned int, unsigned int>> parentMap;
	boost::disjoint_sets<rankMapT, parentMapT> sets(boost::make_assoc_property_map(rankMap), boost::make_assoc_property_map(parentMap));
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < numFGComponents[i]; j++)
			sets.make_set(std::make_pair(i, j));

	vector<vector<unordered_map<unsigned int, unordered_set<unsigned int>>>> corrs(numScenes); //scene 1 -> scene 2 -> scene-1 seg -> scene-2 segs
	for(unsigned int i = 0; i < numScenes; i++) corrs[i].resize(numScenes);

	boost::regex sameRex("sameObj\\(\\w+(\\d+)_(\\d+),\\w+(\\d+)_(\\d+)\\)"), corrRex("corrSelected\\(\\w+(\\d+)_(\\d+),\\w+(\\d+)_(\\d+)\\)");
	ifstream infile((mlndir / "alchemy.out").string());
	ASSERT_ALWAYS(infile);
	string predicate;
	if(inferenceIsMAP)
	{
		while(infile >> predicate)
		{
			boost::smatch match;
			if(boost::regex_match(predicate, match, sameRex))
			{
				const unsigned int scene1 = indexMap[lexical_cast<unsigned int>(match[1])], comp1 = lexical_cast<unsigned int>(match[2]),
					scene2 = indexMap[lexical_cast<unsigned int>(match[3])], comp2 = lexical_cast<unsigned int>(match[4]);
				sets.union_set(std::make_pair(scene1, comp1), std::make_pair(scene2, comp2));
			}
			else if(boost::regex_match(predicate, match, corrRex))
			{
				const unsigned int scene1 = indexMap[lexical_cast<unsigned int>(match[1])], comp1 = lexical_cast<unsigned int>(match[2]),
					scene2 = indexMap[lexical_cast<unsigned int>(match[3])], comp2 = lexical_cast<unsigned int>(match[4]);
				ASSERT_ALWAYS(scene1 < numScenes);
				ASSERT_ALWAYS(scene2 < numScenes);
				corrs[scene1][scene2][comp1].insert(comp2);
			}
			else
			{
				cout << "no match: '" << predicate << "'" << endl;
				getchar();
			}
		}
	}
	else
	{
		double prob;
		while(infile >> predicate >> prob)
		{
			boost::smatch match;
			if(boost::regex_match(predicate, match, sameRex))
			{
				const unsigned int scene1 = lexical_cast<unsigned int>(match[1]), comp1 = lexical_cast<unsigned int>(match[2]),
					scene2 = lexical_cast<unsigned int>(match[3]), comp2 = lexical_cast<unsigned int>(match[4]);
				if(prob > .5) sets.union_set(std::make_pair(scene1, comp1), std::make_pair(scene2, comp2));
			}
			else ASSERT_ALWAYS(false); //TODO
		}
	}
	infile.close();

	/*
	 * get the list of representatives and map them to [0 .. n)
	 */
	boost::unordered_map<pair<unsigned int, unsigned int>, unsigned int> representative2index; //map set IDs into the integer range [0 .. n) for some n
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < numFGComponents[i]; j++)
		{
			const pair<unsigned int, unsigned int> rep = sets.find_set(std::make_pair(i, j));
			if(representative2index.find(rep) == representative2index.end())
			{
				const unsigned int n = representative2index.size(); //ensure this will be computed before operator [] happens, jic
				representative2index[rep] = n;
			}
		}
	cout << "found " << representative2index.size() << " clusters" << endl;

	/*
	 * map cluster ids to component ids
	 */
	vector<unordered_set<pair<unsigned int, unsigned int>>> cluster2comps(representative2index.size());
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < numFGComponents[i]; j++)
			cluster2comps[representative2index[sets.find_set(std::make_pair(i, j))]].insert(std::make_pair(i, j));
	cout << "cluster sizes > 1:";
	for(unsigned int j = 0; j < cluster2comps.size(); j++)
		if(cluster2comps[j].size() > 1)
			cout << ' ' << cluster2comps[j].size() << endl;
	cout << endl;

	/*
	 * visualize
	 */
	const vector<boost::array<unsigned char, 3>> cols = std::move(getDistinguishableColors(representative2index.size() + 1));
	for(unsigned int i = 0; i < numScenes; i++)
	{
		pcl::PointCloud<rgbd::pt> cloud;
		cloud.points.resize(surfel2undersegID[i].size());
		cloud.width = cloud.points.size();
		cloud.height = 1;
		cloud.is_dense = false;
		unsigned int k = 0;
		for(auto j = surfel2undersegID[i].begin(); j != surfel2undersegID[i].end(); j++, k++) //for all foreground surfels
		{
			cloud.points[k].x = scenes[i].surfelCloudPtr->points[(*j).first].x;
			cloud.points[k].y = scenes[i].surfelCloudPtr->points[(*j).first].y;
			cloud.points[k].z = scenes[i].surfelCloudPtr->points[(*j).first].z;
			if(surfel2componentIndex[i].find((*j).first) != surfel2componentIndex[i].end())
			{
				const unsigned int clusterID = representative2index[sets.find_set(std::make_pair(i, surfel2componentIndex[i][(*j).first]))];
				ASSERT_ALWAYS(clusterID < cluster2comps.size());
				cloud.points[k].rgb = rgbd::packRGB(cols[clusterID]);
			}
			else
			{
				cloud.points[k].rgb = rgbd::packRGB(cols.back()); //lump all small clusters into one color for ease of viewing
			}
		}
		rgbd::write_ply_file(cloud, smoutdir / ("fgCompClusters-" + scenes[i].sceneName + ".ply"));
	}

	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < numScenes; j++)
			if(j != i)
			{
				ofstream outfile((smoutdir / ("selectedCorrs-" + scenes[i].sceneName + "-" + scenes[j].sceneName + ".dat")).string());
				ASSERT_ALWAYS(outfile);
				for(auto k = corrs[i][j].begin(); k != corrs[i][j].end(); k++)
					for(auto l = (*k).second.begin(); l != (*k).second.end(); l++)
						outfile << (*k).first << ' ' << *l << endl;
			}

	t.stop("read alchemy outputs");

	return 0;
}
