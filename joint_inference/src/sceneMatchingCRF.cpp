/*
 * sceneMatchingCRF: code to set up and run CRF inference for scene matching
 *
 * Evan Herbst
 * 11 / 4 / 10
 */

#include <cfloat> //DBL_MAX
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include "rgbd_util/timer.h"
#include "rgbd_util/mathUtils.h" //clamp()
#include "rgbd_util/ioUtils.h" //waitKey()
#include "evh_util/multibase_counter.h"
#include "evh_util/cloudSegmentationUtils.h" //makeKDTreeForSegments()
#include "pcl_rgbd/cloudSearchTrees.h"
#include "scene_matching/segmentation/foregroundSurfelSegmentation.h"
#include "scene_matching/segmentNeighborSimilarity.h"
#include "scene_matching/joint_inference/sceneMatchingCRF.h"
using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::set;
using std::string;
using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::ostringstream;
using std::cout;
using std::endl;
using boost::lexical_cast;
using rgbd::eigen::VectorXd;
using rgbd::eigen::Vector3f;

template <typename T>
std::ostream& operator << (std::ostream& out, const vector<T>& v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, " "));
	return out;
}

template <typename T>
std::ostream& operator << (std::ostream& out, const set<T>& v)
{
	std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, " "));
	return out;
}

/*
 * clique of a single assoc var
 */
class assocCliqueEnergyFunc
{
	public:

		static const unsigned int NUM_FEATS = 2;

		assocCliqueEnergyFunc(unordered_map<unsigned int, vector<unsigned int>>& segCorrs, const VectorXd& w)
		: legalSegCorrs(segCorrs),
		  weights(w)
		{
			ASSERT_ALWAYS(weights.size() == NUM_FEATS);
		}

		double operator () (const unsigned int seg1, const vector<unsigned int>& vals)
		{
			ASSERT_ALWAYS(vals.size() == 1);
			const bool noassoc = (vals[0] == legalSegCorrs[seg1].size());
			VectorXd features(NUM_FEATS);
			features[0] = noassoc;
			features[1] = !noassoc;
			return features.dot(weights);
		}

	private:

		unordered_map<unsigned int, vector<unsigned int>>& legalSegCorrs;

		VectorXd weights;
};

/*
 * clique of two nbring obj ids
 */
class nbrCliqueEnergyFunc
{
	public:

		static const unsigned int NUM_FEATS = 12;

		nbrCliqueEnergyFunc(unordered_map<unsigned int, unordered_set<unsigned int>>& segNbrsWithSimilarNormals,
				unordered_map<unsigned int, unordered_set<unsigned int>>& segNbrsWithConcaveJunction,
				unordered_map<unsigned int, unordered_set<unsigned int>>& segNbrsWithConvexJunction,
				unordered_map<unsigned int, unordered_set<unsigned int>>& likeLikes,
				unordered_map<unsigned int, unordered_set<unsigned int>>& likeDislikes,
				const VectorXd& w)
		: simNbrsMap(segNbrsWithSimilarNormals),
		  concaveNbrsMap(segNbrsWithConcaveJunction),
		  convexNbrsMap(segNbrsWithConvexJunction),
		  likeLikeMap(likeLikes),
		  likeDislikeMap(likeDislikes),
		  weights(w)
		{
			ASSERT_ALWAYS(weights.size() == NUM_FEATS);
		}

		double operator () (const unsigned int seg1, const unsigned int seg2,
			unordered_map<unsigned int, unsigned int>& objIDVarMap,
			const set<unsigned int>& varIDs, const vector<unsigned int>& vals)
		{
			return 0; //TODO

			unsigned int label1 = UINT_MAX, label2 = UINT_MAX;
			auto l = vals.begin();
			for(auto m = varIDs.begin(); m != varIDs.end(); m++, l++)
				if(*m == objIDVarMap[seg1]) label1 = *l;
				else if(*m == objIDVarMap[seg2]) label2 = *l;
			ASSERT_ALWAYS(label1 != UINT_MAX);
			ASSERT_ALWAYS(label2 != UINT_MAX);

			const bool sameLabel = (label1 == label2);
			VectorXd features(NUM_FEATS);
			features << 1 * sameLabel,
							(simNbrsMap[seg1].find(seg2) != simNbrsMap[seg1].end()) * sameLabel,
							(convexNbrsMap[seg1].find(seg2) != convexNbrsMap[seg1].end()) * sameLabel,
							(concaveNbrsMap[seg1].find(seg2) != concaveNbrsMap[seg1].end()) * sameLabel,
							(likeLikeMap[seg1].find(seg2) != likeLikeMap[seg1].end()) * sameLabel,
							(likeDislikeMap[seg1].find(seg2) != likeDislikeMap[seg1].end()) * sameLabel,
							1 * !sameLabel,
							(simNbrsMap[seg1].find(seg2) != simNbrsMap[seg1].end()) * !sameLabel,
							(convexNbrsMap[seg1].find(seg2) != convexNbrsMap[seg1].end()) * !sameLabel,
							(concaveNbrsMap[seg1].find(seg2) != concaveNbrsMap[seg1].end()) * !sameLabel,
							(likeLikeMap[seg1].find(seg2) != likeLikeMap[seg1].end()) * !sameLabel,
							(likeDislikeMap[seg1].find(seg2) != likeDislikeMap[seg1].end()) * !sameLabel;
			return features.dot(weights);
		}

	private:

		unordered_map<unsigned int, unordered_set<unsigned int>>& simNbrsMap;
		unordered_map<unsigned int, unordered_set<unsigned int>>& concaveNbrsMap;
		unordered_map<unsigned int, unordered_set<unsigned int>>& convexNbrsMap;
		unordered_map<unsigned int, unordered_set<unsigned int>>& likeLikeMap;
		unordered_map<unsigned int, unordered_set<unsigned int>>& likeDislikeMap;

		VectorXd weights;
};

/*
 * clique of scene-1 seg/scene-2 seg/seg-1 assoc
 */
class corrCliqueEnergyFunc
{
	public:

		static const unsigned int NUM_FEATS = 3;

		corrCliqueEnergyFunc(unordered_map<unsigned int, vector<unsigned int>>& segCorrs, const VectorXd& w)
		: legalSegCorrs(segCorrs),
		  weights(w)
		{
			ASSERT_ALWAYS(weights.size() == NUM_FEATS);
		}

		double operator () (const unsigned int sc1seg, const unsigned int sc2seg,
			unordered_map<unsigned int, unsigned int>& objIDVarMap, unordered_map<unsigned int, unsigned int>& assocVarMap,
			const set<unsigned int>& varIDs, const vector<unsigned int>& vals)
		{
			return 0; //TODO

			unsigned int label1 = UINT_MAX, label2 = UINT_MAX, assoc1val = UINT_MAX;
			auto l = vals.begin();
			for(auto m = varIDs.begin(); m != varIDs.end(); m++, l++)
				if(*m == objIDVarMap[sc1seg]) label1 = *l;
				else if(*m == objIDVarMap[sc2seg]) label2 = *l;
				else if(*m == assocVarMap[sc1seg]) assoc1val = *l;
			ASSERT_ALWAYS(label1 != UINT_MAX);
			ASSERT_ALWAYS(label2 != UINT_MAX);
			ASSERT_ALWAYS(assoc1val <= legalSegCorrs[sc1seg].size());

			const bool noassoc = (assoc1val == legalSegCorrs[sc1seg].size());
			VectorXd features(NUM_FEATS);
			features[0] = noassoc;
			if(!noassoc)
			{
				const bool sameLabel = (label1 == label2);
				const unsigned int assoc1 = legalSegCorrs[sc1seg][assoc1val]; //scene-2 seg index
				features[1] = (assoc1 == sc2seg) && sameLabel;
				features[2] = (assoc1 == sc2seg) && !sameLabel;
			}
			else
			{
				features[1] = 0;
				features[2] = 0;
			}
			return features.dot(weights);
		}

	private:

		unordered_map<unsigned int, vector<unsigned int>>& legalSegCorrs;

		VectorXd weights;
};

/*
 * clique of a nbr pair and two match flags
 */
class matchPairCliqueEnergyFunc
{
	public:

		static const unsigned int NUM_FEATS = 2;

		matchPairCliqueEnergyFunc(unordered_map<unsigned int, unordered_map<unsigned int, unordered_map<unsigned int, unordered_map<unsigned int, double>>>>& segMatchPairXDists,
				unordered_map<unsigned int, unordered_map<unsigned int, unordered_map<unsigned int, unordered_map<unsigned int, double>>>>& segMatchPairAngDists,
				unordered_map<unsigned int, vector<unsigned int>>& segCorrs,
				const VectorXd& w)
		: xDists(segMatchPairXDists),
		  angDists(segMatchPairAngDists),
		  legalSegCorrs(segCorrs),
		  weights(w)
		{
			ASSERT_ALWAYS(weights.size() == NUM_FEATS);
		}

		double operator () (const unsigned int sc1seg1, const unsigned int sc1seg2,
			unordered_map<unsigned int, unsigned int>& objIDVarMap, unordered_map<unsigned int, unsigned int>& assocVarMap,
			const set<unsigned int>& varIDs, const vector<unsigned int>& vals)
		{
			unsigned int label1 = UINT_MAX, label2 = UINT_MAX, assoc1val = UINT_MAX, assoc2val = UINT_MAX;
			auto l = vals.begin();
			for(auto m = varIDs.begin(); m != varIDs.end(); m++, l++)
				if(*m == objIDVarMap[sc1seg1]) label1 = *l;
				else if(*m == objIDVarMap[sc1seg2]) label2 = *l;
				else if(*m == assocVarMap[sc1seg1]) assoc1val = *l;
				else if(*m == assocVarMap[sc1seg2]) assoc2val = *l;
			ASSERT_ALWAYS(label1 != UINT_MAX);
			ASSERT_ALWAYS(label2 != UINT_MAX);
			cout << assoc1val << ' ' << legalSegCorrs[sc1seg1].size() << endl;
			cout << assoc2val << ' ' << legalSegCorrs[sc1seg2].size() << endl;
			ASSERT_ALWAYS(assoc1val <= legalSegCorrs[sc1seg1].size());
			ASSERT_ALWAYS(assoc2val <= legalSegCorrs[sc1seg2].size());

			const bool noassoc1 = (assoc1val == legalSegCorrs[sc1seg1].size()), noassoc2 = (assoc2val == legalSegCorrs[sc1seg2].size());
			const bool sameLabel = (label1 == label2);
			VectorXd features(NUM_FEATS);
			if(/*sameLabel &&*//*TODO*/ !noassoc1 && !noassoc2)
			{
				const unsigned int assoc1 = legalSegCorrs[sc1seg1][assoc1val], assoc2 = legalSegCorrs[sc1seg2][assoc2val]; //scene-2 seg indices
				ASSERT_ALWAYS(xDists[sc1seg1][sc1seg2][assoc1].find(assoc2) != xDists[sc1seg1][sc1seg2][assoc1].end());
				features[0] = xDists[sc1seg1][sc1seg2][assoc1][assoc2];
				features[1] = angDists[sc1seg1][sc1seg2][assoc1][assoc2];
			}
			else
			{
				features[0] = 0;
				features[1] = 0;
			}
			return features.dot(weights);
		}

	private:

		unordered_map<unsigned int, unordered_map<unsigned int, unordered_map<unsigned int, unordered_map<unsigned int, double>>>>& xDists;
		unordered_map<unsigned int, unordered_map<unsigned int, unordered_map<unsigned int, unordered_map<unsigned int, double>>>>& angDists;
		unordered_map<unsigned int, vector<unsigned int>>& legalSegCorrs;

		VectorXd weights;
};

/*
 * likeLikes, likeDislikes: scene -> seg 1 -> seg2
 * segCorrs: scene 1 -> scene 2 -> scene-1 seg -> scene-2 segs it can correspond with
 */
void twoSceneSceneMatchingCRFInference(const vector<boost::shared_ptr<sceneInfo>>& scenes, const unsigned int scene1index, const unsigned int scene2index,
	vector<unordered_map<unsigned int, unordered_set<unsigned int>>>& likeLikes,
	vector<unordered_map<unsigned int, unordered_set<unsigned int>>>& likeDislikes,
	vector<vector<unordered_map<unsigned int, unordered_set<unsigned int>>>>& segCorrs,
	const fs::path& outdir, const string& segFilebase)
{
	const fs::path crfOutdir = outdir / ("crf-" + scenes[scene1index]->sceneName + "-" + scenes[scene2index]->sceneName);
	fs::create_directories(crfOutdir);
	const vector<unsigned int> sceneIndices = {scene1index, scene2index};
	const unsigned int numScenes = 2;
	const vector<unsigned int> numSegments =
	{
		loadNumForegroundSegments(*scenes[sceneIndices[0]], outdir / segFilebase),
		loadNumForegroundSegments(*scenes[sceneIndices[1]], outdir / segFilebase)
	};
//	const unsigned int totNumSegs = numSegments[0] + numSegments[1];
	const unsigned int numObjIDs = 4; //TODO ?

	VectorXd assocCliqueEnergyWeights(2),
		nbrCliqueEnergyWeights(12),
		matchPairCliqueEnergyWeights(2),
		corrCliqueEnergyWeights(3);
	assocCliqueEnergyWeights << .5, 0;
	nbrCliqueEnergyWeights << -.1, -.05, -.05, .05, -1, .25,
									.1, .05, .05, -.05, 1, -.25;
	matchPairCliqueEnergyWeights << 1, .6;
	corrCliqueEnergyWeights << 0, 0, log(1e100);

	/*
	 * put lists of associatable segs into vectors in the same order that the sets have (to get random access)
	 */
	vector<vector<unordered_map<unsigned int, vector<unsigned int>>>> segCorrsVecs(numScenes);
	for(unsigned int i = 0; i < numScenes; i++)
	{
		segCorrsVecs[i].resize(numScenes);
		for(unsigned int j = 0; j < numScenes; j++)
			if(j != i)
				for(auto k = segCorrs[i][j].begin(); k != segCorrs[i][j].end(); k++)
					segCorrsVecs[i][j][(*k).first] = std::vector<unsigned int>((*k).second.begin(), (*k).second.end());
	}

	/*
	 * graph:
	 * - var per seg, giving obj id
	 * - var per plausible match (boolean)
	 * - clique per nbr pair
	 * - clique per plausible match flag + two ids (if selected, ids must match)
	 * - clique per nbr pair + pair of plausible match flags (if same obj & both selected, must be geom. consistent)
	 * - clique per fwd/bkwd pair of plausible match flags (must be neither or both selected)
	 */

	rgbd::timer t;
	const fs::path fastinfInfilepath = crfOutdir / ("fastinf-" + scenes[scene1index]->sceneName + "-" + scenes[scene2index]->sceneName + ".net");
	ofstream outfile(fastinfInfilepath.string());

#define objVarStr(scene, seg) ("objvar" + lexical_cast<string>(scene) + "_" + lexical_cast<string>(seg))
#define assocVarStr(scene1, scene2, seg1) ("assocvar" + lexical_cast<string>(scene1) + "_" + lexical_cast<string>(seg1) + "_" + lexical_cast<string>(scene2))
#define assocCliqueStr(scene, seg) ("assocC" + lexical_cast<string>(scene) + "_" + lexical_cast<string>(seg))
#define nbrCliqueStr(scene, seg1, seg2) ("nbrC" + lexical_cast<string>(scene) + "_" + lexical_cast<string>(seg1) + "_" + lexical_cast<string>(seg2))
#define corrCliqueStr(scene1, scene2, seg1, seg2) ("corrC" + lexical_cast<string>(scene1) + "_" + lexical_cast<string>(seg1) + "_" + lexical_cast<string>(scene2) + "_" + lexical_cast<string>(seg2))
#define matchPairCliqueStr(scene1, scene2, sc1seg1, sc1seg2) ("matchPairC" + lexical_cast<string>(scene1) + "_" + lexical_cast<string>(scene2) + "_" + lexical_cast<string>(sc1seg1) + "_" + lexical_cast<string>(sc1seg2))

	//varName  \t  var number of assignments
	outfile << "@Variables" << endl;
	unsigned int m = 0; //fastinf var index
	vector<unordered_map<unsigned int, unsigned int>> objIDVarMap(numScenes); //scene -> seg -> var id
	vector<unordered_map<unsigned int, unsigned int>> assocVarMap(numScenes); //scene -> seg -> var id
	vector<unsigned int> allVarCardinalities; //indexed by var index
	for(unsigned int i = 0; i < numScenes; i++)
	{
		/*
		 * load structures
		 */

		unordered_map<unsigned int, unsigned int> surfel2segIndex1;
		unsigned int numSegments1;
		vector<set<unsigned int>> seg2surfelIndices1;
		vector<Vector3f> segmentCentroids1;
		unordered_map<unsigned int, unordered_map<unsigned int, float>> minCompSpatialDist1;
		unordered_map<unsigned int, unordered_set<unsigned int>> segNbrs1;
		std::tie(surfel2segIndex1, numSegments1, seg2surfelIndices1, segmentCentroids1, minCompSpatialDist1, segNbrs1) =
			std::move(loadForegroundSegments(*scenes[sceneIndices[i]], outdir / segFilebase));

		for(unsigned int j = 0; j < numSegments[i]; j++)
		{
			/*
			 * don't create a var if it won't participate in any cliques; fastinf's results printing chokes on such vars
			 */
			if(!segNbrs1[j].empty() || !segCorrs[sceneIndices[i]][sceneIndices[!i]][j].empty())
			{
				allVarCardinalities.push_back(numObjIDs);
				outfile << objVarStr(sceneIndices[i], j) << '\t' << allVarCardinalities.back() << endl;
				objIDVarMap[i][j] = m++;
			}

			/*
			 * TODO remove cliques for assoc vars w/ no legal corrs, to make inference faster?
			 */
			allVarCardinalities.push_back(segCorrs[sceneIndices[i]][sceneIndices[!i]][j].size() + 1);
			outfile << assocVarStr(sceneIndices[i], sceneIndices[!i], j) << '\t' << allVarCardinalities.back() << endl;
			assocVarMap[i][j] = m++;
		}
	}
	outfile << "@End" << endl;
	const unsigned int numVars = m;
	cout << numVars << " vars" << endl;

	unordered_map<unsigned int, set<unsigned int>> var2cliques; //var id -> ids of all cliques it's in
	unsigned int c = 0; //fastinf clique index
{
	for(unsigned int i = 0; i < numScenes; i++)
	{
		/*
		 * load structures
		 */

		unordered_map<unsigned int, unsigned int> surfel2segIndex1;
		unsigned int numSegments1;
		vector<set<unsigned int>> seg2surfelIndices1;
		vector<Vector3f> segmentCentroids1;
		unordered_map<unsigned int, unordered_map<unsigned int, float>> minCompSpatialDist1;
		unordered_map<unsigned int, unordered_set<unsigned int>> segNbrs1;
		std::tie(surfel2segIndex1, numSegments1, seg2surfelIndices1, segmentCentroids1, minCompSpatialDist1, segNbrs1) =
			std::move(loadForegroundSegments(*scenes[sceneIndices[i]], outdir / segFilebase));

		for(unsigned int j = 0; j < numSegments[i]; j++)
		{
			var2cliques[assocVarMap[sceneIndices[i]][j]].insert(c);
			c++;
		}

		for(unsigned int j = 0; j < numSegments[i]; j++)
			for(auto k = segNbrs1[j].begin(); k != segNbrs1[j].end(); k++)
				if(*k > j) //only list each once
				{
					var2cliques[objIDVarMap[sceneIndices[i]][j]].insert(c);
					var2cliques[objIDVarMap[sceneIndices[i]][*k]].insert(c);
					c++;
				}

		for(unsigned int j = 0; j < numSegments[i]; j++)
			for(auto k = segNbrs1[j].begin(); k != segNbrs1[j].end(); k++)
			{
				var2cliques[objIDVarMap[sceneIndices[i]][j]].insert(c);
				var2cliques[objIDVarMap[sceneIndices[i]][*k]].insert(c);
				var2cliques[assocVarMap[sceneIndices[i]][j]].insert(c);
				var2cliques[assocVarMap[sceneIndices[i]][*k]].insert(c);
				c++;
			}
	}

	for(unsigned int j = 0; j < numSegments[0]; j++)
		for(auto l = segCorrs[sceneIndices[0]][sceneIndices[1]][j].begin(); l != segCorrs[sceneIndices[0]][sceneIndices[1]][j].end(); l++)
		{
			var2cliques[objIDVarMap[sceneIndices[0]][j]].insert(c);
			var2cliques[objIDVarMap[sceneIndices[1]][*l]].insert(c);
			var2cliques[assocVarMap[sceneIndices[0]][j]].insert(c);
			c++;
		}
}
	const unsigned int numCliques = c;
	cout << numCliques << " cliques" << endl;

#define ADD_CLIQUES_FOR_VAR(varID) cliqueIndices.insert(var2cliques[varID].begin(), var2cliques[varID].end())
	ostringstream outcliq, outmeas;
	c = 0;
{
	for(unsigned int i = 0; i < numScenes; i++)
	{
		/*
		 * load structures
		 */

		unordered_map<unsigned int, unsigned int> surfel2segIndex1;
		unsigned int numSegments1;
		vector<set<unsigned int>> seg2surfelIndices1;
		vector<Vector3f> segmentCentroids1;
		unordered_map<unsigned int, unordered_map<unsigned int, float>> minCompSpatialDist1;
		unordered_map<unsigned int, unordered_set<unsigned int>> segNbrs1;
		std::tie(surfel2segIndex1, numSegments1, seg2surfelIndices1, segmentCentroids1, minCompSpatialDist1, segNbrs1) =
			std::move(loadForegroundSegments(*scenes[sceneIndices[i]], outdir / segFilebase));

		unordered_map<unsigned int, unordered_set<unsigned int>> segNbrsWithSimilarNormals, segNbrsWithConcaveJunction, segNbrsWithConvexJunction;
		std::tie(segNbrsWithSimilarNormals, segNbrsWithConcaveJunction, segNbrsWithConvexJunction) =
			std::move(getNeighboringSegmentSimilarities(*scenes[sceneIndices[i]], numSegments1, surfel2segIndex1, segmentCentroids1, segNbrs1, outdir / "segNbrSims"));

		/*
		 * compute some evidence
		 */
		unordered_map<unsigned int, unordered_map<unsigned int, unordered_map<unsigned int, unordered_map<unsigned int, double>>>> segMatchPairXDists, segMatchPairAngDists;
	{
		/*
		 * k-d tree of foreground
		 */
		vector<unsigned int> allSegIDs1(numSegments1);
		for(unsigned int k = 0; k < numSegments1; k++) allSegIDs1[k] = k;
		boost::shared_ptr<kdtree2> fgKDTree1;
		vector<unsigned int> treeIndex2surfelIndex1;
		std::tie(fgKDTree1, treeIndex2surfelIndex1) = makeKDTreeForSegments(*scenes[sceneIndices[i]]->surfelCloudPtr, seg2surfelIndices1, allSegIDs1);

		/*
		 * load structures
		 */
		unordered_map<unsigned int, unsigned int> surfel2segIndex2;
		unsigned int numSegments2;
		vector<set<unsigned int>> seg2surfelIndices2;
		vector<Vector3f> segmentCentroids2;
		unordered_map<unsigned int, unordered_map<unsigned int, float>> minCompSpatialDist2;
		unordered_map<unsigned int, unordered_set<unsigned int>> segNbrs2;
		std::tie(surfel2segIndex2, numSegments2, seg2surfelIndices2, segmentCentroids2, minCompSpatialDist2, segNbrs2) =
			std::move(loadForegroundSegments(*scenes[sceneIndices[!i]], outdir / segFilebase));

		/*
		 * k-d tree of foreground
		 */
		vector<unsigned int> allSegIDs2(numSegments2);
		for(unsigned int k = 0; k < numSegments2; k++) allSegIDs2[k] = k;
		boost::shared_ptr<kdtree2> fgKDTree2;
		vector<unsigned int> treeIndex2surfelIndex2;
		std::tie(fgKDTree2, treeIndex2surfelIndex2) = makeKDTreeForSegments(*scenes[sceneIndices[!i]]->surfelCloudPtr, seg2surfelIndices2, allSegIDs2);

		for(unsigned int k = 0; k < numSegments1; k++)
			for(unsigned int m = 0; m < numSegments1; m++)
			{
				const float scene1intersegDist = (segmentCentroids1[k] - segmentCentroids1[m]).norm();

				unsigned int scene1seg1bestNbr, scene1seg2bestNbr; //surfel indices for seg centroids
			{
				const vector<float> qpt = {segmentCentroids1[k].x(), segmentCentroids1[k].y(), segmentCentroids1[k].z()};
				kdtree2_result_vector result;
				fgKDTree1->n_nearest(qpt, 1, result);
				scene1seg1bestNbr = treeIndex2surfelIndex1[result[0].idx];
			}
			{
				const vector<float> qpt = {segmentCentroids1[m].x(), segmentCentroids1[m].y(), segmentCentroids1[m].z()};
				kdtree2_result_vector result;
				fgKDTree1->n_nearest(qpt, 1, result);
				scene1seg2bestNbr = treeIndex2surfelIndex1[result[0].idx];
			}
				const Vector3f normal11 = rgbd::ptNormal2eigen<Vector3f>(scenes[sceneIndices[i]]->surfelCloudPtr->points[scene1seg1bestNbr]),
					normal12 = rgbd::ptNormal2eigen<Vector3f>(scenes[sceneIndices[i]]->surfelCloudPtr->points[scene1seg2bestNbr]);
				const float scene1normalAngle = acos(normal11.dot(normal12));

				for(auto l = segCorrs[sceneIndices[i]][sceneIndices[!i]][k].begin(); l != segCorrs[sceneIndices[i]][sceneIndices[!i]][k].end(); l++)
					for(auto n = segCorrs[sceneIndices[i]][sceneIndices[!i]][m].begin(); n != segCorrs[sceneIndices[i]][sceneIndices[!i]][m].end(); n++)
						if(m != k) //different src segs     // || *l != *n)
						{
							const float scene2intersegDist = (segmentCentroids2[*l] - segmentCentroids2[*n]).norm();
							const float ddist = fabs(scene2intersegDist - scene1intersegDist);

							unsigned int scene2seg1bestNbr, scene2seg2bestNbr; //surfel indices for seg centroids
						{
							const vector<float> qpt = {segmentCentroids2[*l].x(), segmentCentroids2[*l].y(), segmentCentroids2[*l].z()};
							kdtree2_result_vector result;
							fgKDTree2->n_nearest(qpt, 1, result);
							scene2seg1bestNbr = treeIndex2surfelIndex2[result[0].idx];
						}
						{
							const vector<float> qpt = {segmentCentroids2[*n].x(), segmentCentroids2[*n].y(), segmentCentroids2[*n].z()};
							kdtree2_result_vector result;
							fgKDTree2->n_nearest(qpt, 1, result);
							scene2seg2bestNbr = treeIndex2surfelIndex2[result[0].idx];
						}
							const Vector3f normal21 = rgbd::ptNormal2eigen<Vector3f>(scenes[sceneIndices[!i]]->surfelCloudPtr->points[scene2seg1bestNbr]),
								normal22 = rgbd::ptNormal2eigen<Vector3f>(scenes[sceneIndices[!i]]->surfelCloudPtr->points[scene2seg2bestNbr]);
							float scene2normalAngle = acos(clamp(-1.0f, 1.0f, normal21.dot(normal22)));
							if(scene2normalAngle > scene1normalAngle + M_PI) scene2normalAngle -= 2 * M_PI;
							else if(scene2normalAngle < scene1normalAngle - M_PI) scene2normalAngle += 2 * M_PI;
							const float dangle = fabs(scene2normalAngle - scene1normalAngle);
							ASSERT_ALWAYS(!isinf(ddist));
							ASSERT_ALWAYS(!isnan(ddist));
							ASSERT_ALWAYS(!isinf(dangle));
							ASSERT_ALWAYS(!isnan(dangle));
							segMatchPairXDists[k][m][*l][*n] = ddist;
							segMatchPairAngDists[k][m][*l][*n] = dangle;
						}
			}
	}

		assocCliqueEnergyFunc assocCliqueEnergy(segCorrsVecs[sceneIndices[i]][sceneIndices[!i]], assocCliqueEnergyWeights);
		for(unsigned int j = 0; j < numSegments[i]; j++)
		{
			const string cliqueStr = assocCliqueStr(sceneIndices[i], j);

			set<unsigned int> varIDs; //20101109: I think fastinf wants the vars in index order
			varIDs.insert(assocVarMap[sceneIndices[i]][j]);
			outcliq << cliqueStr << '\t' << varIDs.size() << '\t' << varIDs << '\t';
			/*
			 * list nbring cliques
			 */
			set<unsigned int> cliqueIndices;
			for(auto n = varIDs.begin(); n != varIDs.end(); n++) ADD_CLIQUES_FOR_VAR((*n));
			cliqueIndices.erase(c);
			outcliq << cliqueIndices.size() << '\t' << cliqueIndices << endl;

			vector<unsigned int> varCardinalities(varIDs.size());
			unsigned int o = 0;
			for(auto n = varIDs.begin(); n != varIDs.end(); n++, o++) varCardinalities[o] = allVarCardinalities[*n];
			outmeas << "m_" << cliqueStr << '\t' << varCardinalities.size() << '\t' << varCardinalities << '\t';
			for(multibase_counter<unsigned int> valsIter(varCardinalities); !valsIter.done(); valsIter++)
			{
				outmeas << exp(-assocCliqueEnergy(j, *valsIter)) << ' ';
			}
			outmeas << endl;

			c++;
		}

		nbrCliqueEnergyFunc nbrCliqueEnergy(segNbrsWithSimilarNormals, segNbrsWithConcaveJunction, segNbrsWithConvexJunction,
				likeLikes[sceneIndices[i]], likeDislikes[sceneIndices[i]], nbrCliqueEnergyWeights);
		for(unsigned int j = 0; j < numSegments[i]; j++)
			for(auto k = segNbrs1[j].begin(); k != segNbrs1[j].end(); k++)
				if(*k > j) //only list each once
				{
					const string cliqueStr = nbrCliqueStr(sceneIndices[i], j, *k);

					set<unsigned int> varIDs; //20101109: I think fastinf wants the vars in index order
					varIDs.insert(objIDVarMap[sceneIndices[i]][j]);
					varIDs.insert(objIDVarMap[sceneIndices[i]][*k]);
					outcliq << cliqueStr << '\t' << varIDs.size() << '\t' << varIDs << '\t';
					/*
					 * list nbring cliques
					 */
					set<unsigned int> cliqueIndices;
					for(auto n = varIDs.begin(); n != varIDs.end(); n++) ADD_CLIQUES_FOR_VAR((*n));
					cliqueIndices.erase(c);
					outcliq << cliqueIndices.size() << '\t' << cliqueIndices << endl;

					vector<unsigned int> varCardinalities(varIDs.size());
					unsigned int o = 0;
					for(auto n = varIDs.begin(); n != varIDs.end(); n++, o++) varCardinalities[o] = allVarCardinalities[*n];
					outmeas << "m_" << cliqueStr << '\t' << varCardinalities.size() << '\t' << varCardinalities << '\t';
					for(multibase_counter<unsigned int> valsIter(varCardinalities); !valsIter.done(); valsIter++)
					{
						outmeas << exp(-nbrCliqueEnergy(j, *k, objIDVarMap[sceneIndices[i]], varIDs, *valsIter)) << ' ';
					}
					outmeas << endl;

					c++;
				}

		matchPairCliqueEnergyFunc matchPairCliqueEnergy(segMatchPairXDists, segMatchPairAngDists, segCorrsVecs[sceneIndices[i]][sceneIndices[!i]], matchPairCliqueEnergyWeights);
		for(unsigned int j = 0; j < numSegments[i]; j++)
			for(auto k = segNbrs1[j].begin(); k != segNbrs1[j].end(); k++)
			{
				const string cliqueStr = matchPairCliqueStr(sceneIndices[i], sceneIndices[!i], j, *k);

				set<unsigned int> varIDs; //20101109: I think fastinf wants the vars in index order
				varIDs.insert(objIDVarMap[sceneIndices[i]][j]);
				varIDs.insert(objIDVarMap[sceneIndices[i]][*k]);
				varIDs.insert(assocVarMap[sceneIndices[i]][j]);
				varIDs.insert(assocVarMap[sceneIndices[i]][*k]);
				outcliq << cliqueStr << '\t' << varIDs.size() << '\t' << varIDs << '\t';
				/*
				 * list nbring cliques
				 */
				set<unsigned int> cliqueIndices;
				for(auto n = varIDs.begin(); n != varIDs.end(); n++) ADD_CLIQUES_FOR_VAR((*n));
				cliqueIndices.erase(c);
				outcliq << cliqueIndices.size() << '\t' << cliqueIndices << endl;

				vector<unsigned int> varCardinalities(varIDs.size());
				unsigned int o = 0;
				for(auto n = varIDs.begin(); n != varIDs.end(); n++, o++) varCardinalities[o] = allVarCardinalities[*n];
				outmeas << "m_" << cliqueStr << '\t' << varCardinalities.size() << '\t' << varCardinalities << '\t';
				for(multibase_counter<unsigned int> valsIter(varCardinalities); !valsIter.done(); valsIter++)
				{
					outmeas << exp(-matchPairCliqueEnergy(j, *k, objIDVarMap[sceneIndices[i]], assocVarMap[sceneIndices[i]], varIDs, *valsIter)) << ' ';
				}
				outmeas << endl;

				c++;
			}
	}

	corrCliqueEnergyFunc corrCliqueEnergy(segCorrsVecs[sceneIndices[0]][sceneIndices[1]], corrCliqueEnergyWeights);
	for(unsigned int j = 0; j < numSegments[0]; j++)
		for(auto l = segCorrs[sceneIndices[0]][sceneIndices[1]][j].begin(); l != segCorrs[sceneIndices[0]][sceneIndices[1]][j].end(); l++)
		{
			const string cliqueStr = corrCliqueStr(sceneIndices[0], sceneIndices[1], j, *l);

			set<unsigned int> varIDs; //20101109: I think fastinf wants the vars in index order
			varIDs.insert(objIDVarMap[sceneIndices[0]][j]);
			varIDs.insert(objIDVarMap[sceneIndices[1]][*l]);
			varIDs.insert(assocVarMap[sceneIndices[0]][j]);
			outcliq << cliqueStr << '\t' << varIDs.size() << '\t' << varIDs << '\t';
			/*
			 * list nbring cliques
			 */
			set<unsigned int> cliqueIndices;
			for(auto n = varIDs.begin(); n != varIDs.end(); n++) ADD_CLIQUES_FOR_VAR((*n));
			cliqueIndices.erase(c);
			outcliq << cliqueIndices.size() << '\t' << cliqueIndices << endl;

			vector<unsigned int> varCardinalities(varIDs.size());
			unsigned int o = 0;
			for(auto n = varIDs.begin(); n != varIDs.end(); n++, o++) varCardinalities[o] = allVarCardinalities[*n];
			outmeas << "m_" << cliqueStr << '\t' << varCardinalities.size() << '\t' << varCardinalities << '\t';
			for(multibase_counter<unsigned int> valsIter(varCardinalities); !valsIter.done(); valsIter++)
			{
				outmeas << exp(-corrCliqueEnergy(j, *l, objIDVarMap[sceneIndices[0]], assocVarMap[sceneIndices[0]], varIDs, *valsIter)) << ' ';
			}
			outmeas << endl;

			c++;
		}
}
#undef ADD_CLIQUES_FOR_VAR

	//clique name \t  number of vars in clique \t the list of vars in clique \t  number of neighbors \t list of neighbouring cliques
	outfile << "@Cliques" << endl;
	outfile << outcliq.str();
	outfile << "@End" << endl;

	//measureName \t  number of vars in measure \t the vars card \t  table of potentials
	//the table is ordered when the assign advances like a number i.e : 000 001 010 011 100 ...
	outfile << "@Measures" << endl;
	outfile << outmeas.str();
	outfile << "@End" << endl;

	//num of clique \t num of measure
	outfile << "@CliqueToMeasure" << endl;
	for(unsigned int i = 0; i < numCliques; i++)
		outfile << i << '\t' << i << endl;
	outfile << "@End" << endl;

	outfile.close();
	t.stop("write fastinf inputs");
	waitKey();

	const fs::path fastinfOutfilepath = crfOutdir / ("fastinf-" + scenes[scene1index]->sceneName + "-" + scenes[scene2index]->sceneName + ".out");
	ostringstream outstr;
	outstr << "/home/eherbst/libs/fastInf/build/bin/infer -i " << fastinfInfilepath << " -x + -It 1e-5 -m 0 -Om " << fastinfOutfilepath;
	const string cmd = outstr.str();
	cout << "to run: '" << cmd << "'" << endl;
#if 1
	system(cmd.c_str());
#else
	getchar(); //let the user run it if desired
#endif

	/*
	 * parse crf outputs
	 */
	ifstream infile(fastinfOutfilepath.string() + ".success");
	ASSERT_ALWAYS(infile);
	string line;
	getline(infile, line);
	ASSERT_ALWAYS(line[0] == '#');
	vector<vector<double>> probsByVarID(numVars);
	for(unsigned int j = 0; j < numVars; j++)
	{
		getline(infile, line);
		ASSERT_ALWAYS(infile);
		istringstream instr(line);
		unsigned int varNum;
		instr >> varNum;
		ASSERT_ALWAYS(varNum == j);
		double p;
		while(instr >> p) probsByVarID[j].push_back(p);
//		cout << " " << probsByVarID[j].size() << " probs" << endl;
	}
	infile.close();

	/*
	 * interpret crf outputs
	 */
	vector<unordered_map<unsigned int, unsigned int>> objIDValMap(numScenes); //scene -> seg -> var id
	vector<unordered_map<unsigned int, unsigned int>> assocValMap(numScenes); //scene -> seg -> var id
	for(unsigned int i = 0; i < numScenes; i++)
	{
		for(auto j = objIDVarMap[i].begin(); j != objIDVarMap[i].end(); j++)
		{
			const unsigned int varID = (*j).second;
			unsigned int maxIndex;
			double maxProb = -DBL_MAX;
			for(unsigned int k = 0; k < probsByVarID[varID].size(); k++)
				if(probsByVarID[varID][k] > maxProb)
				{
					maxProb = probsByVarID[varID][k];
					maxIndex = k;
				}
			objIDValMap[i][(*j).first] = maxIndex;
		}

		for(auto j = assocVarMap[i].begin(); j != assocVarMap[i].end(); j++)
		{
			const unsigned int varID = (*j).second;
			unsigned int maxIndex;
			double maxProb = -DBL_MAX;
			for(unsigned int k = 0; k < probsByVarID[varID].size(); k++)
				if(probsByVarID[varID][k] > maxProb)
				{
					maxProb = probsByVarID[varID][k];
					maxIndex = k;
				}
			assocValMap[i][(*j).first] = maxIndex;
		}
	}

	/*
	 * visualize
	 */

#if 0
	const vector<boost::array<unsigned char, 3>> cols = std::move(getDistinguishableColors(numObjIDs));
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
#endif

{
	ofstream outfile((outdir / ("selectedCorrs-" + scenes[sceneIndices[0]]->sceneName + "-" + scenes[sceneIndices[1]]->sceneName + ".dat")).string());
	ASSERT_ALWAYS(outfile);
	for(auto k = assocValMap[sceneIndices[0]].begin(); k != assocValMap[sceneIndices[0]].end(); k++)
		if((*k).second != segCorrsVecs[sceneIndices[0]][sceneIndices[1]][(*k).first].size())
			outfile << (*k).first << ' ' << segCorrsVecs[sceneIndices[0]][sceneIndices[1]][(*k).first][(*k).second] << endl;
}

}
