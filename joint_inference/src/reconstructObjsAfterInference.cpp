/*
 * reconstructObjsAfterInference: create object models from scenes and the result of scene matching
 *
 * Evan Herbst
 * 9 / 6 / 10
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
#include "rgbd_util/serializationUtils.h"
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
#include "point_cloud_icp/registration/icp_utility.h"
#include "point_cloud_icp/registration/icp_combined.h"
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
using rgbd::eigen::Affine3f;
using rgbd::eigen::AngleAxisf;
using rgbd::eigen::Translation3f;

/*
 * arguments: mln outdir, # scenes, scene names, scene #s to look for
 */
int main(int argc, char* argv[])
{
	ros::Time::init();
	ASSERT_ALWAYS(argc >= 2);

	/********************************************************************************************************
	 * read datasets
	 */

	unsigned int _ = 1;
	const fs::path mlndir(argv[_++]), smoutdir = mlndir.parent_path(), datasetDir = smoutdir.parent_path();
	const unsigned int numScenes = lexical_cast<unsigned int>(argv[_++]);
	ASSERT_ALWAYS(numScenes == 2); //or update this file

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
	vector<unordered_map<unsigned int, unordered_set<unsigned int>>> seg2surfelIndices(numScenes);
	vector<unsigned int> numFGComponents(numScenes);
	boost::format surf2compFilepathFmt((smoutdir / "segsForXforms-%1%.out").string()); //args: scene name
	for(unsigned int i = 0; i < numScenes; i++)
	{
		const fs::path surf2compFilepath((surf2compFilepathFmt % scenes[i].sceneName).str());
		ASSERT_ALWAYS(fs::exists(surf2compFilepath));
		{
			deserializeStructures(surf2compFilepath.string(), surfel2componentIndex[i]);

			numFGComponents[i] = 0;
			for(auto j = surfel2componentIndex[i].begin(); j != surfel2componentIndex[i].end(); j++)
				if((*j).second >= numFGComponents[i])
					numFGComponents[i] = (*j).second + 1;

			for(auto j = surfel2componentIndex[i].begin(); j != surfel2componentIndex[i].end(); j++)
				seg2surfelIndices[i][(*j).second].insert((*j).first);
		}
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
			cout << "corr " << scene1 << ' ' << comp1 << ' ' << scene2 << ' ' << comp2 << endl;
		}
		else
		{
			cout << "no match: '" << predicate << "'" << endl;
			getchar();
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
	 * reconstruct
	 */
	for(unsigned int m = 0; m < cluster2comps.size(); m++)
		if(cluster2comps[m].size() >= 5/* TODO parameterize */)
		{
			cout << "cluster " << m << endl;
			for(unsigned int i = 0; i < numScenes; i++)
				for(unsigned int j = 0; j < numScenes; j++)
					if(j > i)
					{
						vector<pair<int, int>> scenePairCorrs, inlierCorrs; //indices into segment lists
						for(auto k = corrs[i][j].begin(); k != corrs[i][j].end(); k++)
						{
							ASSERT_ALWAYS((*k).first < numFGComponents[i]);
							const pair<unsigned int, unsigned int> seg1(i, (*k).first);
							cout << " on seg " << i << ' ' << (*k).first << endl;
							const unsigned int index = representative2index[sets.find_set(seg1)];
							cout << "  cluster " << index << endl;
							ASSERT_ALWAYS(cluster2comps[index].find(seg1) != cluster2comps[index].end());
							if(cluster2comps[m].find(seg1) != cluster2comps[m].end())
								for(auto l = (*k).second.begin(); l != (*k).second.end(); l++)
								{
									ASSERT_ALWAYS(*l < numFGComponents[j]);
									const pair<unsigned int, unsigned int> seg2(j, *l);
									const unsigned int index = representative2index[sets.find_set(seg2)];
									cout << "    cluster " << index << endl;
									if(cluster2comps[m].find(seg2) != cluster2comps[m].end())
										scenePairCorrs.push_back(std::make_pair((*k).first, *l));
								}
						}
						for(auto k = corrs[j][i].begin(); k != corrs[j][i].end(); k++)
						{
							ASSERT_ALWAYS((*k).first < numFGComponents[j]);
							const pair<unsigned int, unsigned int> seg1(j, (*k).first);
							if(cluster2comps[m].find(seg1) != cluster2comps[m].end())
								for(auto l = (*k).second.begin(); l != (*k).second.end(); l++)
								{
									ASSERT_ALWAYS(*l < numFGComponents[i]);
									const pair<unsigned int, unsigned int> seg2(i, *l);
									if(cluster2comps[m].find(seg2) != cluster2comps[m].end())
										scenePairCorrs.push_back(std::make_pair((*k).first, *l));
								}
						}
						cout << "ransac incorrs: " << scenePairCorrs.size() << endl;
						if(scenePairCorrs.size() >= 3)
						{
							Affine3f ransacXform;
							const bool success = registration::runRANSAC(segmentCentroids[i], segmentCentroids[j], scenePairCorrs, .035/* inlier dist */, ransacXform, inlierCorrs, 40000/* # iters */, 3, 0, false/* distance downweighting */, false/* verbose */);
							if(!success) cout << "ransac failed" << endl;
							else
							{

							cout << "ransac took " << scenePairCorrs.size() << " corrs to " << inlierCorrs.size() << endl;

							unsigned int scene1cloudSize = 0, scene2cloudSize = 0;
							for(auto k = cluster2comps[m].begin(); k != cluster2comps[m].end(); k++)
								if((*k).first == i)
								{
									scene1cloudSize += seg2surfelIndices[i][(*k).second].size();
								}
								else if((*k).first == j)
								{
									scene2cloudSize += seg2surfelIndices[j][(*k).second].size();
								}
							pcl::PointCloud<rgbd::surfelPt> scene1cloud, scene2cloud;
							scene1cloud.points.resize(scene1cloudSize);
							scene1cloud.width = scene1cloud.points.size();
							scene1cloud.height = 1;
							scene1cloud.is_dense = false;
							scene2cloud.points.resize(scene2cloudSize);
							scene2cloud.width = scene2cloud.points.size();
							scene2cloud.height = 1;
							scene2cloud.is_dense = false;
							unsigned int n1 = 0, n2 = 0; //indices into sceneNcloud
							for(auto k = cluster2comps[m].begin(); k != cluster2comps[m].end(); k++)
								if((*k).first == i)
								{
									for(auto l = seg2surfelIndices[i][(*k).second].begin(); l != seg2surfelIndices[i][(*k).second].end(); l++, n1++)
									{
										ASSERT_ALWAYS(*l < scenes[i].surfelCloudPtr->points.size());
										scene1cloud.points[n1] = scenes[i].surfelCloudPtr->points[*l];
									}
								}
								else if((*k).first == j)
								{
									for(auto l = seg2surfelIndices[j][(*k).second].begin(); l != seg2surfelIndices[j][(*k).second].end(); l++, n2++)
									{
										ASSERT_ALWAYS(*l < scenes[j].surfelCloudPtr->points.size());
										scene2cloud.points[n2] = scenes[j].surfelCloudPtr->points[*l];
									}
								}
							ASSERT_ALWAYS(n1 == scene1cloudSize);
							ASSERT_ALWAYS(n2 == scene2cloudSize);
							registration::ICPCloudPairParams cpParams;
							cpParams.errType = registration::ICP_ERR_POINT_TO_PLANE;
							cpParams.max_distance = -1; //TODO tweak
							cpParams.use_average_point_error = true;
							cpParams.outlier_percentage = .65; //TODO tweak
							boost::shared_ptr<registration::ICPCloudPair> cloudPair(new registration::ICPCloudPair(cpParams, scene1cloud, scene2cloud));
							registration::ICPCombinedParams icpParams;
							icpParams.optimizer = registration::OPTIMIZER_LEVMAR;
							icpParams.max_lm_rounds = 50;
							icpParams.max_icp_rounds = 10;
							registration::ICPCombined icp;
							icp.setParams(icpParams);
							icp.addCloudPair(cloudPair);
							icp.setInitialTransform(ransacXform);
							Affine3f icpXform;
							icp.runICP(icpXform);

							pcl::PointCloud<rgbd::surfelPt> resultCloud;
							resultCloud.points.resize(scene1cloudSize + scene2cloudSize);
							resultCloud.width = resultCloud.points.size();
							resultCloud.height = 1;
							resultCloud.is_dense = false;
							unsigned int q = 0; //index into resultCloud.points
							for(auto k = cluster2comps[m].begin(); k != cluster2comps[m].end(); k++)
								if((*k).first == i)
								{
									for(auto l = seg2surfelIndices[i][(*k).second].begin(); l != seg2surfelIndices[i][(*k).second].end(); l++, q++)
									{
										const Vector3f xformedPt = icpXform * rgbd::ptX2eigen<Vector3f>(scenes[i].surfelCloudPtr->points[*l]);
										resultCloud.points[q] = scenes[i].surfelCloudPtr->points[*l];
										resultCloud.points[q].x = xformedPt.x();
										resultCloud.points[q].y = xformedPt.y();
										resultCloud.points[q].z = xformedPt.z();
									}
								}
								else if((*k).first == j)
								{
									for(auto l = seg2surfelIndices[j][(*k).second].begin(); l != seg2surfelIndices[j][(*k).second].end(); l++, q++)
										resultCloud.points[q] = scenes[j].surfelCloudPtr->points[*l];
								}
							rgbd::write_ply_file(resultCloud, "recon" + lexical_cast<string>(m) + ".ply");

							//TODO reconstruct

							}
						}
						//TODO what if only 2 or 1 corrs?
					}

			//(sba for all scenes)
		}

#if 0
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
#endif

	t.stop("read alchemy outputs");

	return 0;
}
