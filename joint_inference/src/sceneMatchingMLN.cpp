/*
 * sceneMatchingMLN: code to set up and run MLN inference for scene matching
 *
 * Evan Herbst
 * 11 / 4 / 10
 */

#include <cassert>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include "rgbd_util/timer.h"
#include "rgbd_util/mathUtils.h" //sqr()
#include "evh_util/cloudSegmentationUtils.h" //makeKDTreeForSegments()
#include "scene_matching/segmentation/foregroundSurfelSegmentation.h"
#include "scene_matching/segmentNeighborSimilarity.h"
#include "scene_matching/joint_inference/sceneMatchingMLN.h"
using std::vector;
using std::unordered_set;
using std::string;
using std::ofstream;
using std::ostringstream;
using std::cout;
using std::endl;
using boost::lexical_cast;

/*
 * likeLikes, likeDislikes: scene -> seg 1 -> seg2
 * segCorrs: scene 1 -> scene 2 -> scene-1 seg -> scene-2 segs it can correspond with
 */
void twoSceneSceneMatchingMLNInference(const vector<boost::shared_ptr<sceneInfo>>& scenes, const unsigned int scene1index, const unsigned int scene2index,
	vector<unordered_map<unsigned int, unordered_set<unsigned int>>>& likeLikes,
	vector<unordered_map<unsigned int, unordered_set<unsigned int>>>& likeDislikes,
	vector<vector<unordered_map<unsigned int, unordered_set<unsigned int>>>>& segCorrs,
	const fs::path& outdir, const std::string& mlnSegFilebase)
{
	const fs::path mlnOutdir = outdir / ("mln-" + scenes[scene1index]->sceneName + "-" + scenes[scene2index]->sceneName);
	fs::create_directories(mlnOutdir);
	const vector<unsigned int> sceneIndices = {scene1index, scene2index};

	rgbd::timer t;
	ofstream outfile;

	/*
	write mln file
		define types and constants
	*/
//#define sceneIDStr(scene) ("S" + lexical_cast<string>(scene))
#define compIDStr(scene, seg) ("C" + lexical_cast<string>(scene) + "_" + lexical_cast<string>(seg))
//#define corrIDStr(scene1, seg1, scene2, seg2) ("R" + lexical_cast<string>(scene1) + "_" + lexical_cast<string>(seg1) + "_" + lexical_cast<string>(scene2) + "_" + lexical_cast<string>(seg2))
	outfile.open((mlnOutdir / "domain.mln").string());
	ASSERT_ALWAYS(outfile);

#if 0
	outfile << "scene = { ";
	for(auto i = sceneIndices.begin(); i != sceneIndices.end(); i++)
		outfile << sceneIDStr(*i) << ' ';
	outfile << "}" << endl;
#endif

	outfile << "comp = { ";
	for(auto i = sceneIndices.begin(); i != sceneIndices.end(); i++)
	{
		/*
		 * load structures
		 */
		const unsigned int numSegments1 = loadNumForegroundSegments(*scenes[*i], outdir / mlnSegFilebase);

		for(unsigned int j = 0; j < numSegments1; j++)
			outfile << compIDStr(*i, j) << ' ';
	}
	outfile << "}" << endl;

//#define USE_CORRS
#define USE_CORRS_SEG_VARS
#ifdef USE_CORRS
	outfile << "corr = { ";
	for(auto i = sceneIndices.begin(); i != sceneIndices.end(); i++)
		for(auto j = sceneIndices.begin(); j != sceneIndices.end(); j++)
			if(j != i)
				for(auto k = allLegalSegCorrs[*i][*j].begin(); k != allLegalSegCorrs[*i][*j].end(); k++)
					for(auto l = (*k).second.begin(); l != (*k).second.end(); l++)
						outfile << corrIDStr(*i, (*k).first, *j, *l) << ' ';
	outfile << "}" << endl;
#endif

	outfile << endl;

	/*
	write mln file
		define predicates
		define formulae
	*/

	outfile << "sameScene(comp,comp)" << endl;
	outfile << "segNbrs(comp,comp)" << endl; //for all pairs of close segments
	outfile << "segNbrsAndSimilarNormals(comp,comp)" << endl; //close segments with similar normals near their boundary
	outfile << "segNbrsConcave(comp,comp)" << endl; //close segments forming a concave surface
	outfile << "segNbrsConvex(comp,comp)" << endl; //close segments forming a convex surface
	outfile << "likeLike(comp,comp)" << endl; //w/in scenes, do segs like/dislike the same candidate xforms from ransac?
	outfile << "likeDislike(comp,comp)" << endl;

#ifdef USE_CORRS
	//outfile << "comp corrSrc(corr)" << endl; //Marc says don't use functions
	outfile << "corrHasSrc(corr,comp)" << endl;
	outfile << "corrHasDest(corr,comp)" << endl;
	outfile << "corrsSameObj(corr,corr)" << endl;
	outfile << "corrsSameSrc(corr,corr)" << endl;
	outfile << "corrsSameDest(corr,corr)" << endl;
	outfile << "corrsSameScenes(corr,corr)" << endl;
	//outfile << "corrsDistCompatible(corr,corr)" << endl;
	outfile << "corrsDistIncompatible(corr,corr)" << endl;
	outfile << "corrSelected(corr)" << endl;
#elif defined(USE_CORRS_SEG_VARS)
	outfile << "corrProposed(comp,comp)" << endl;
	outfile << "corrSelected(comp,comp)" << endl;
	outfile << "corrsDistIncompatible(comp,comp,comp,comp)" << endl; //(a -> b, c -> d)
	outfile << "corrsAngleIncompatible(comp,comp,comp,comp)" << endl; //(a -> b, c -> d)
	outfile << "sameSrcCorrsIncompatible(comp,comp,comp)" << endl; //(a -> b, a -> c)
#endif

#ifdef USE_OBJ_IDS
	outfile << "obj(comp,comp)" << endl; //the second one is a stand-in for an obj id
#else
	outfile << "sameObj(comp,comp)" << endl;
#endif
	outfile << endl;

#ifdef USE_OBJ_IDS
	outfile << 1 << " obj(s,o)" << endl;
	outfile << .11 << " sameScene(s,t) ^ obj(s,o) => !obj(t,o)" << endl;
	outfile << .13 << " !sameScene(s,t) ^ obj(s,o) => !obj(t,o)" << endl;
	outfile << .02 << " segNbrs(s,t) ^ obj(s,o) => obj(t,o)" << endl;
	outfile << .09 << " segNbrsAndSimilarNormals(s,t) ^ obj(s,o) => obj(t,o)" << endl;
	outfile << .03 << " segNbrsConvex(s,t) ^ obj(s,o) => obj(t,o)" << endl;
	outfile << .09 << " segNbrsConcave(s,t) ^ obj(s,o) => !obj(t,o)" << endl;
	outfile << 1 << " likeLike(s,t) ^ obj(s,o) => obj(t,o)" << endl;
	outfile << .35 << " likeDislike(s,t) ^ obj(s,o) => !obj(t,o)" << endl;
	//don't need transitivity; happens automatically
#else

#if 0 //works pretty well for 3objs3/4 (which has no ransac xforms), 20100824, and for 3objs1/3 with ransac xforms only, 20100825
	outfile << .1 << " sameScene(s,t) => !sameObj(s,t)" << endl;
	outfile << .12 << " !sameScene(s,t) => !sameObj(s,t)" << endl;
	outfile << .05 << " segNbrs(s,t) => sameObj(s,t)" << endl;
	outfile << 1 << " likeLike(s,t) => sameObj(s,t)" << endl;
	outfile << .3 << " likeDislike(s,t) => !sameObj(s,t)" << endl;
	outfile << "sameObj(s,s)." << endl;
	outfile << "sameObj(s,t) => sameObj(t,s)." << endl;
	outfile << "segNbrs(s,t) ^ sameScene(s,u) ^ sameObj(s,t) ^ sameObj(t,u) => sameObj(s,u)." << endl;
#endif
#if 0 //works pretty well for icraA4/5, and incorporates corrs, 20100904
	0.09 sameScene(s,t) => !sameObj(s,t)
	0.13 !sameScene(s,t) => !sameObj(s,t)
	0.1 segNbrs(s,t) => sameObj(s,t)
	0.1 segNbrsAndSimilarNormals(s,t) => sameObj(s,t)
	0.04 segNbrsConvex(s,t) => sameObj(s,t)
	0.06 segNbrsConcave(s,t) => !sameObj(s,t)
	1 likeLike(s,t) => sameObj(s,t)
	sameObj(s,t) => sameObj(t,s).
	sameObj(s,t) ^ sameObj(t,u) => sameObj(s,u).
	!corrProposed(s,t) => !corrSelected(s,t).
	0.0001 corrProposed(s,t) => corrSelected(s,t)
	corrSelected(s,t) => corrSelected(t,s).
	corrSelected(s,t) => sameObj(s,t).
	50 corrProposed(s,t) ^ corrProposed(s,u) ^ !segNbrs(t,u) => !(corrSelected(s,t) ^ corrSelected(s,u))
	100 corrsDistIncompatible(s,u,t,v) => !(sameObj(s,t) ^ corrSelected(s,u) ^ corrSelected(t,v))
#endif

#if 0
	outfile << .1 << " !sameObj(s,t)" << endl;
#else //if want different weights for same & different scenes (but harder to set weights than in the one-formula case)
	outfile << .09 << " sameScene(s,t) => !sameObj(s,t)" << endl;
	outfile << .13 << " !sameScene(s,t) => !sameObj(s,t)" << endl;
#endif
	outfile << .08 << " segNbrs(s,t) => sameObj(s,t)" << endl;
	outfile << .03 << " segNbrsAndSimilarNormals(s,t) => sameObj(s,t)" << endl;
	outfile << .04 << " segNbrsConvex(s,t) => sameObj(s,t)" << endl; //nbring segs form a convex corner
	outfile << .06 << " segNbrsConcave(s,t) => !sameObj(s,t)" << endl; //nbring segs form a concave corner

	outfile << 1 << " likeLike(s,t) => sameObj(s,t)" << endl;
	//outfile << .25 << " likeDislike(s,t) => !sameObj(s,t)" << endl; //not proving very helpful

	outfile << "sameObj(s,t) => sameObj(t,s)." << endl;
//		outfile << "sameScene(s,t) ^ sameScene(s,u) ^ sameObj(s,t) ^ sameObj(t,u) => sameObj(s,u)." << endl; //the precondition keeps the explosion of ground clauses down
	/*
	 * if not using MAP inference, this makes it very very slow (not quite as bad if use -saTemperature 10, -saRatio 0)
	 *
	 * if doing MAP, this actually uses less memory w/o any precondition than with sameScene(s,t) ^ sameScene(s,u) conjuncted in
	 */
	outfile << "sameObj(s,t) ^ sameObj(t,u) => sameObj(s,u)." << endl;
#endif

#ifdef USE_CORRS
	/*
	 * prior that we like there to be some corrs
	 */
	outfile << .001 << " corrSelected(c)" << endl;

	outfile << "corrHasSrc(c, s) ^ corrHasDest(c, t) ^ corrSelected(c) => sameObj(s, t)." << endl;
	//may not need uniqueness constraints if geometric consistency gives the same info
	outfile << "corrsSameSrc(c, d) ^ corrsSameScenes(c, d) => !(corrSelected(c) ^ corrSelected(d))." << endl; //unique mapping per src seg per dest scene (c != d implied)
	//outfile << "corrsSameDest(c, d) ^ corrsSameScenes(c, d) => !(corrSelected(c) ^ corrSelected(d))." << endl; //unique mapping per dest seg (c != d implied)
#if 0 //incompatible is faster than compatible
	outfile << "corrsSameScenes(c, d) ^ !corrsSameSrc(c, d) ^ !corrsDistCompatible(c, d) => !(srcsSameObj(c, d) ^ corrSelected(c) ^ corrSelected(d))." << endl;
#elif 0 //orders of magnitude slower than the multi-formula way
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) ^ corrsDistIncompatible(c, d) => !(sameObj(s, t) ^ corrSelected(c) ^ corrSelected(d))." << endl;
#else
	outfile << "corrsDistIncompatible(c, d) => !(corrsSameObj(c, d) ^ corrSelected(c) ^ corrSelected(d))." << endl;
#if 1
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) => (sameObj(s, t) <=> corrsSameObj(c, d))." << endl;
	outfile << "corrHasDest(c, s) ^ corrHasDest(d, t) => (sameObj(s, t) <=> corrsSameObj(c, d))." << endl;
#else
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) ^ sameObj(s, t) => corrsSameObj(c, d)." << endl;
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) ^ corrsSameObj(c, d) => sameObj(s, t)." << endl;
	outfile << "corrHasDest(c, s) ^ corrHasDest(d, t) ^ sameObj(s, t) => corrsSameObj(c, d)." << endl;
	outfile << "corrHasDest(c, s) ^ corrHasDest(d, t) ^ corrsSameObj(c, d) => sameObj(s, t)." << endl;
#endif
	/* much slower, probably because corrSelected isn't evidence
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) ^ corrSelected(c) ^ corrSelected(d) ^ sameObj(s, t) => srcsSameObj(c, d)." << endl;
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) ^ corrSelected(c) ^ corrSelected(d) ^ srcsSameObj(c, d) => sameObj(s, t)." << endl;
	*/
#endif
#elif defined(USE_CORRS_SEG_VARS)
#ifdef USE_OBJ_IDS
	outfile << "corrSelected(s,t) => corrSelected(t,s)." << endl;
	outfile << "corrSelected(s,t) ^ obj(s,o) => obj(t,o)." << endl;
	outfile << "!corrProposed(s,t) => !corrSelected(s,t)." << endl;
	/*
	 * prior that we like there to be some corrs
	 */
	outfile << .0001 << " corrProposed(s,t) => corrSelected(s,t)" << endl;

	//outfile << "corrProposed(s,t) ^ corrProposed(s,u) ^ sameScene(t,u) => !(corrSelected(s,t) ^ corrSelected(s,u))." << endl; //unique mapping per src seg per dest scene (t != u implied)
	outfile << 1 << " !(t = u) ^ corrProposed(s,t) ^ corrProposed(s,u) ^ !segNbrs(t,u) => !(corrSelected(s,t) ^ corrSelected(s,u))" << endl; //mappings per src seg per dest scene must be close

	outfile << 2 << " corrsDistIncompatible(s,u,t,v) => !(obj(s,o) ^ obj(t,o) ^ corrSelected(s,u) ^ corrSelected(t,v))" << endl;
#else
	outfile << "corrSelected(s,t) => corrSelected(t,s)." << endl;
	outfile << "corrSelected(s,t) => sameObj(s,t)." << endl;
	outfile << "!corrProposed(s,t) => !corrSelected(s,t)." << endl;
	/*
	 * prior that we like there to be some corrs
	 */
	outfile << .0001 << " corrProposed(s,t) => corrSelected(s,t)" << endl;

	//outfile << 100 << " !(t = u) ^ corrProposed(s,t) ^ corrProposed(s,u) ^ !segNbrs(t,u) => !(corrSelected(s,t) ^ corrSelected(s,u))" << endl; //mappings per src seg per dest scene must be close
	outfile << 100 << " sameSrcCorrsIncompatible(s,t,u) => !(corrSelected(s,t) ^ corrSelected(s,u))" << endl;

	outfile << 20 << " corrsDistIncompatible(s,u,t,v) => !(sameObj(s,t) ^ corrSelected(s,u) ^ corrSelected(t,v))" << endl;
	outfile << 20 << " corrsAngleIncompatible(s,u,t,v) => !(sameObj(s,t) ^ corrSelected(s,u) ^ corrSelected(t,v))" << endl;
#endif
#endif

	outfile.close();

	/*
	write evidence
	*/
	outfile.open((mlnOutdir / "evidence.db").string());
	ASSERT_ALWAYS(outfile);

	/*
	 * per-scene evidence
	 */
	for(auto i = sceneIndices.begin(); i != sceneIndices.end(); i++)
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
			std::move(loadForegroundSegments(*scenes[*i], outdir / mlnSegFilebase));

		/*
		 * sameScene
		 */
		for(unsigned int j = 0; j < numSegments1; j++)
			for(unsigned int k = 0; k < numSegments1; k++)
				if(j != k)
					outfile << "sameScene(" << compIDStr(*i, j) << "," << compIDStr(*i, k) << ")" << endl;

		/********************************************************************************************************
		 * evidence from neighboring-segment similarity measures
		 */

		unordered_map<unsigned int, unordered_set<unsigned int>> segNbrsWithSimilarNormals, segNbrsWithConcaveJunction, segNbrsWithConvexJunction;
		std::tie(segNbrsWithSimilarNormals, segNbrsWithConcaveJunction, segNbrsWithConvexJunction) =
			std::move(getNeighboringSegmentSimilarities(*scenes[*i], numSegments1, surfel2segIndex1, segmentCentroids1, segNbrs1, outdir / "segNbrSims"));

		for(auto j = segNbrs1.begin(); j != segNbrs1.end(); j++)
		{
			for(auto k = (*j).second.begin(); k != (*j).second.end(); k++)
			{
				outfile << "segNbrs(" << compIDStr(*i, (*j).first) << "," << compIDStr(*i, *k) << ")" << endl;

				if(segNbrsWithSimilarNormals[(*j).first].find(*k) != segNbrsWithSimilarNormals[(*j).first].end())
				{
					outfile << "segNbrsAndSimilarNormals(" << compIDStr(*i, (*j).first) << "," << compIDStr(*i, *k) << ")" << endl;
				}

				if(segNbrsWithConcaveJunction[(*j).first].find(*k) != segNbrsWithConcaveJunction[(*j).first].end())
				{
					outfile << "segNbrsConcave(" << compIDStr(*i, (*j).first) << "," << compIDStr(*i, *k) << ")" << endl;
				}
				else if(segNbrsWithConvexJunction[(*j).first].find(*k) != segNbrsWithConvexJunction[(*j).first].end())
				{
					outfile << "segNbrsConvex(" << compIDStr(*i, (*j).first) << "," << compIDStr(*i, *k) << ")" << endl;
				}
			}
		}

		/********************************************************************************************************
		 * hack: components not close in space are different objs
		 * TODO remove
		 */
		for(unsigned int j = 0; j < numSegments1 - 1; j++)
			for(unsigned int k = j + 1; k < numSegments1; k++)
				if(minCompSpatialDist1[j][k] > sqr(.55)) //dino is 52 cm long
					outfile << "!sameObj(" << compIDStr(*i, j) << "," << compIDStr(*i, k) << ")" << endl;

		/********************************************************************************************************
		 * like same xforms
		 */
		for(auto j = likeLikes[*i].begin(); j != likeLikes[*i].end(); j++)
			for(auto k = (*j).second.begin(); k != (*j).second.end(); k++)
				outfile << "likeLike(" << compIDStr(*i, (*j).first) << "," << compIDStr(*i, *k) << ")" << endl;
		for(auto j = likeDislikes[*i].begin(); j != likeDislikes[*i].end(); j++)
			for(auto k = (*j).second.begin(); k != (*j).second.end(); k++)
				outfile << "likeDislike(" << compIDStr(*i, (*j).first) << "," << compIDStr(*i, *k) << ")" << endl;
	}

	/*
	 * interscene correspondence
	 */
	for(auto i = sceneIndices.begin(); i != sceneIndices.end(); i++)
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
			std::move(loadForegroundSegments(*scenes[*i], outdir / mlnSegFilebase));

		/*
		 * k-d tree of foreground
		 */
		vector<unsigned int> allSegIDs1(numSegments1);
		for(unsigned int k = 0; k < numSegments1; k++) allSegIDs1[k] = k;
		boost::shared_ptr<kdtree2> fgKDTree1;
		vector<unsigned int> treeIndex2surfelIndex1;
		std::tie(fgKDTree1, treeIndex2surfelIndex1) = makeKDTreeForSegments(*scenes[*i]->surfelCloudPtr, seg2surfelIndices1, allSegIDs1);

		for(auto j = sceneIndices.begin(); j != sceneIndices.end(); j++)
			if(j != i)
			{
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
					std::move(loadForegroundSegments(*scenes[*j], outdir / mlnSegFilebase));

				/*
				 * k-d tree of foreground
				 */
				vector<unsigned int> allSegIDs2(numSegments2);
				for(unsigned int k = 0; k < numSegments2; k++) allSegIDs2[k] = k;
				boost::shared_ptr<kdtree2> fgKDTree2;
				vector<unsigned int> treeIndex2surfelIndex2;
				std::tie(fgKDTree2, treeIndex2surfelIndex2) = makeKDTreeForSegments(*scenes[*j]->surfelCloudPtr, seg2surfelIndices2, allSegIDs2);

#ifdef USE_CORRS
				for(auto k = allLegalSegCorrs[*i][*j].begin(); k != allLegalSegCorrs[*i][*j].end(); k++) //for each corr from scene i
					for(auto l = (*k).second.begin(); l != (*k).second.end(); l++)
					{
						outfile << "corrHasSrc(" << corrIDStr(*i, (*k).first, *j, *l) << "," << compIDStr(*i, (*k).first) << ")" << endl;
						outfile << "corrHasDest(" << corrIDStr(*i, (*k).first, *j, *l) << "," << compIDStr(*j, *l) << ")" << endl;

						for(auto m = (*k).second.begin(); m != (*k).second.end(); m++)
							if(*m != *l/* different corrs */)
								outfile << "corrsSameSrc(" << corrIDStr(*i, (*k).first, *j, *l) << "," << corrIDStr(*i, (*k).first, *j, *m) << ")" << endl;

						for(auto m = allLegalSegCorrs[*i][*j].begin(); m != allLegalSegCorrs[*i][*j].end(); m++) //for each corr from scene i
							for(auto n = (*m).second.begin(); n != (*m).second.end(); n++)
							{
								if((*k).first != (*m).first/* different corrs */ && *n == *l)
									outfile << "corrsSameDest(" << corrIDStr(*i, (*k).first, *j, *l) << "," << corrIDStr(*i, (*m).first, *j, *n) << ")" << endl;

								if((*m).first != (*k).first || *n != *l)/* different corrs */
									outfile << "corrsSameScenes(" << corrIDStr(*i, (*k).first, *j, *l) << "," << corrIDStr(*i, (*m).first, *j, *n) << ")" << endl;
							}
					}
#elif defined(USE_CORRS_SEG_VARS)
				for(auto k = segCorrs[*i][*j].begin(); k != segCorrs[*i][*j].end(); k++) //for each corr from scene i
					for(auto l = (*k).second.begin(); l != (*k).second.end(); l++)
						outfile << "corrProposed(" << compIDStr(*i, (*k).first) << "," << compIDStr(*j, *l) << ")" << endl;
#endif

				for(auto k = segCorrs[*i][*j].begin(); k != segCorrs[*i][*j].end(); k++)
					for(auto l = (*k).second.begin(); l != (*k).second.end(); l++)
						for(auto m = (*k).second.begin(); m != (*k).second.end(); m++)
							if(*m != *l)
								if(segNbrs2[*m].find(*l) == segNbrs2[*m].end())
									outfile << "sameSrcCorrsIncompatible(" << compIDStr(*i, (*k).first) << "," << compIDStr(*j, *l) << "," << compIDStr(*j, *m) << ")" << endl;

//					ofstream outfile3((outdir / ("ddists-" + scenes[i].sceneName + "-" + scenes[j].sceneName + ".mat")).string());
//					ASSERT_ALWAYS(outfile3);
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
						const Vector3f normal11 = rgbd::ptNormal2eigen<Vector3f>(scenes[*i]->surfelCloudPtr->points[scene1seg1bestNbr]),
							normal12 = rgbd::ptNormal2eigen<Vector3f>(scenes[*i]->surfelCloudPtr->points[scene1seg2bestNbr]);
						const float scene1normalAngle = acos(normal11.dot(normal12));

						for(auto l = segCorrs[*i][*j][k].begin(); l != segCorrs[*i][*j][k].end(); l++)
							for(auto n = segCorrs[*i][*j][m].begin(); n != segCorrs[*i][*j][m].end(); n++)
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
									const Vector3f normal21 = rgbd::ptNormal2eigen<Vector3f>(scenes[*j]->surfelCloudPtr->points[scene2seg1bestNbr]),
										normal22 = rgbd::ptNormal2eigen<Vector3f>(scenes[*j]->surfelCloudPtr->points[scene2seg2bestNbr]);
									float scene2normalAngle = acos(normal21.dot(normal22));
									if(scene2normalAngle > scene1normalAngle + M_PI) scene2normalAngle -= 2 * M_PI;
									else if(scene2normalAngle < scene1normalAngle - M_PI) scene2normalAngle += 2 * M_PI;
									const float dangle = fabs(scene2normalAngle - scene1normalAngle);

//										outfile3 << ddist << ' ';
#ifdef USE_CORRS
									if(ddist < .02/*.8 * avgSegDiameter*//* TODO parameterize */) //TODO move the numerical bit above this section
										;//outfile << "corrsDistCompatible(" << corrIDStr(i, k, j, *l) << "," << corrIDStr(i, m, j, *n) << ")" << endl;
									else
										outfile << "corrsDistIncompatible(" << corrIDStr(*i, k, *j, *l) << "," << corrIDStr(*i, m, *j, *n) << ")" << endl;
#elif defined(USE_CORRS_SEG_VARS)
									//.02 for icraA45/mln45
									if(ddist < .03/* TODO parameterize */) //TODO move the numerical bit above this section
										;//outfile << "corrsDistCompatible(" << corrIDStr(i, k, j, *l) << "," << corrIDStr(i, m, j, *n) << ")" << endl;
									else
										outfile << "corrsDistIncompatible(" << compIDStr(*i, k) << "," << compIDStr(*j, *l) << "," << compIDStr(*i, m) << "," << compIDStr(*j, *n) << ")" << endl;

									if(dangle < .35/* TODO parameterize */)
										;
									else
										outfile << "corrsAngleIncompatible(" << compIDStr(*i, k) << "," << compIDStr(*j, *l) << "," << compIDStr(*i, m) << "," << compIDStr(*j, *n) << ")" << endl;
#endif
								}
					}
			}
	}

	outfile.close();
	t.stop("write alchemy inputs");

	/*
	run alchemy

	does -lazyLowState help w/ speed? -- no (still true 20100902)

	don't need -lazy if using -memLimit? -- do need it

	don't use memLimit; Marc says it's deprecated due to being unreliable
	*/
	string queryFormulae = "sameObj";
#ifdef USE_CORRS
	queryFormulae += ",corrSelected";
#elif defined(USE_CORRS_SEG_VARS)
	queryFormulae += ",corrSelected";
#endif
	ostringstream outstr;
	outstr << "/home/eherbst/software/alchemy/bin/infer -i " << (mlnOutdir / "domain.mln") << " -e " << (mlnOutdir / "evidence.db") << " -r " << (mlnOutdir / "alchemy.out") << " -q " << queryFormulae << " -lazy";
#if 1 //not sure whether necessary, but works on icraA4/5, 20100904
	outstr << " -breakHardClauses";
#endif
	/*
	 * if using transitivity, add: -saTemperature 10 -saRatio 0
	 * (instead of tens of minutes w/ transitivity or 5-10 sec w/o, MC-SAT will take 1.5 min)
	 *
	 * for MAP inference, add -m -mwsMaxSteps N (default N is 1000000)
	 */
	const bool inferenceIsMAP = true;
	const bool usingTransitivity = true;
	if(inferenceIsMAP) outstr << " -m -mwsMaxSteps 30000";
	else
	{
		outstr << " -maxSteps 1000";
		if(usingTransitivity) outstr << " -saTemperature 10 -saRatio 0";
	}
	cout << "to run: '" << outstr.str() << "'" << endl;
}

/****************************************************************************************************************************
 *
 *
 *
 *
 ***************************************************************************************************************************/

/*
 * likeLikes, likeDislikes: scene -> seg 1 -> seg2
 * segCorrs: scene 1 -> scene 2 -> scene-1 seg -> scene-2 segs it can correspond with
 */
void multisceneSceneMatchingMLNInference(const vector<boost::shared_ptr<sceneInfo>>& scenes,
	vector<unordered_map<unsigned int, unordered_set<unsigned int>>>& likeLikes,
	vector<unordered_map<unsigned int, unordered_set<unsigned int>>>& likeDislikes,
	vector<vector<unordered_map<unsigned int, unordered_set<unsigned int>>>>& segCorrs,
	const fs::path& outdir, const string& mlnSegFilebase)
{
	const unsigned int numScenes = scenes.size();

	rgbd::timer t;
	ofstream outfile;

	/*
	write mln file
		define types and constants
	*/
#define sceneIDStr(scene) ("S" + lexical_cast<string>(scene))
#define compIDStr(scene, seg) ("C" + lexical_cast<string>(scene) + "_" + lexical_cast<string>(seg))
#define corrIDStr(scene1, seg1, scene2, seg2) ("R" + lexical_cast<string>(scene1) + "_" + lexical_cast<string>(seg1) + "_" + lexical_cast<string>(scene2) + "_" + lexical_cast<string>(seg2))
	outfile.open((outdir / "constants.mln").string());
	ASSERT_ALWAYS(outfile);

	outfile << "scene = { ";
	for(unsigned int i = 0; i < numScenes; i++)
		outfile << sceneIDStr(i) << ' ';
	outfile << "}" << endl;

	outfile << "comp = { ";
	for(unsigned int i = 0; i < numScenes; i++)
	{
		/*
		 * load structures
		 */
		const unsigned int numSegments1 = loadNumForegroundSegments(*scenes[i], outdir / mlnSegFilebase);

		for(unsigned int j = 0; j < numSegments1; j++)
			outfile << compIDStr(i, j) << ' ';
	}
	outfile << "}" << endl;

	outfile << "corr = { ";
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < numScenes; j++)
			if(j != i)
				for(auto k = segCorrs[i][j].begin(); k != segCorrs[i][j].end(); k++)
					for(auto l = (*k).second.begin(); l != (*k).second.end(); l++)
						outfile << corrIDStr(i, (*k).first, j, *l) << ' ';
	outfile << "}" << endl;

	outfile.close();

	/*
	write mln file
		define predicates
		define formulae
	*/
	outfile.open((outdir / "domain.mln").string());
	ASSERT_ALWAYS(outfile);

	outfile << "#include \"" << outdir << "/constants.mln\"" << endl;
	outfile << endl;
	outfile << "sameScene(comp,comp)" << endl;
	outfile << "segNbrs(comp,comp)" << endl; //for all pairs of close segments
	outfile << "segNbrsAndSimilarNormals(comp,comp)" << endl; //close segments with similar normals near the boundary
	outfile << "segNbrsConcave(comp,comp)" << endl; //close segments forming a concave surface
	outfile << "segNbrsConvex(comp,comp)" << endl; //close segments forming a convex surface
	outfile << "likeLike(comp,comp)" << endl; //w/in scenes, do segs like/dislike the same candidate xforms from ransac?
	outfile << "likeDislike(comp,comp)" << endl;

//#define USE_CORRS
#ifdef USE_CORRS
	//outfile << "comp corrSrc(corr)" << endl; //Marc says don't use functions
	outfile << "corrHasSrc(corr,comp)" << endl;
	outfile << "corrHasDest(corr,comp)" << endl;
	outfile << "corrsSameObj(corr,corr)" << endl;
	outfile << "corrsSameSrc(corr,corr)" << endl;
	outfile << "corrsSameDest(corr,corr)" << endl;
	outfile << "corrsSameScenes(corr,corr)" << endl;
	//outfile << "corrsDistCompatible(corr,corr)" << endl;
	outfile << "corrsDistIncompatible(corr,corr)" << endl;
	outfile << "corrSelected(corr)" << endl;
#endif

	outfile << "sameObj(comp,comp)" << endl;
	outfile << endl;

#if 0 //works pretty well for 3objs3/4 (which has no ransac xforms), 20100824, and for 3objs1/3 with ransac xforms only, 20100825
	outfile << .1 << " sameScene(s,t) => !sameObj(s,t)" << endl;
	outfile << .12 << " !sameScene(s,t) => !sameObj(s,t)" << endl;
	outfile << .05 << " segNbrs(s,t) => sameObj(s,t)" << endl;
	outfile << 1 << " likeLike(s,t) => sameObj(s,t)" << endl;
	outfile << .3 << " likeDislike(s,t) => !sameObj(s,t)" << endl;
	outfile << "sameObj(s,s)." << endl;
	outfile << "sameObj(s,t) => sameObj(t,s)." << endl;
	outfile << "segNbrs(s,t) ^ sameScene(s,u) ^ sameObj(s,t) ^ sameObj(t,u) => sameObj(s,u)." << endl;
#endif

#if 0
	outfile << .1 << " !sameObj(s,t)" << endl;
#else //if want different weights for same & different scenes (but harder to set weights than in the one-formula case)
	outfile << .11 << " sameScene(s,t) => !sameObj(s,t)" << endl;
	outfile << .13 << " !sameScene(s,t) => !sameObj(s,t)" << endl;
#endif
	outfile << .02 << " segNbrs(s,t) => sameObj(s,t)" << endl;
	outfile << .09 << " segNbrsAndSimilarNormals(s,t) => sameObj(s,t)" << endl;
	//concave/convex get tacked onto the existing evidence from segNbrs, so weights should be smaller than that formula's
	outfile << .03 << " segNbrsConvex(s,t) => sameObj(s,t)" << endl; //nbring segs form a convex corner
	outfile << .09 << " segNbrsConcave(s,t) => !sameObj(s,t)" << endl; //nbring segs form a concave corner

	outfile << 1 << " likeLike(s,t) => sameObj(s,t)" << endl;
	outfile << .35 << " likeDislike(s,t) => !sameObj(s,t)" << endl;

#ifdef USE_CORRS
	/*
	 * prior that we like there to be some corrs
	 */
	outfile << .001 << " corrSelected(c)" << endl;

	outfile << "corrHasSrc(c, s) ^ corrHasDest(c, t) ^ corrSelected(c) => sameObj(s, t)." << endl;
	//may not need uniqueness constraints if geometric consistency gives the same info
	outfile << "corrsSameSrc(c, d) ^ corrsSameScenes(c, d) => !(corrSelected(c) ^ corrSelected(d))." << endl; //unique mapping per src seg (c != d implied)
	outfile << "corrsSameDest(c, d) ^ corrsSameScenes(c, d) => !(corrSelected(c) ^ corrSelected(d))." << endl; //unique mapping per dest seg (c != d implied)
#if 0 //incompatible is faster than compatible
	outfile << "corrsSameScenes(c, d) ^ !corrsSameSrc(c, d) ^ !corrsDistCompatible(c, d) => !(srcsSameObj(c, d) ^ corrSelected(c) ^ corrSelected(d))." << endl;
#elif 0 //orders of magnitude slower than the multi-formula way
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) ^ corrsDistIncompatible(c, d) => !(sameObj(s, t) ^ corrSelected(c) ^ corrSelected(d))." << endl;
#else
	outfile << "corrsDistIncompatible(c, d) => !(corrsSameObj(c, d) ^ corrSelected(c) ^ corrSelected(d))." << endl;
#if 1
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) => (sameObj(s, t) <=> corrsSameObj(c, d))." << endl;
	outfile << "corrHasDest(c, s) ^ corrHasDest(d, t) => (sameObj(s, t) <=> corrsSameObj(c, d))." << endl;
#else
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) ^ sameObj(s, t) => corrsSameObj(c, d)." << endl;
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) ^ corrsSameObj(c, d) => sameObj(s, t)." << endl;
	outfile << "corrHasDest(c, s) ^ corrHasDest(d, t) ^ sameObj(s, t) => corrsSameObj(c, d)." << endl;
	outfile << "corrHasDest(c, s) ^ corrHasDest(d, t) ^ corrsSameObj(c, d) => sameObj(s, t)." << endl;
#endif
	/* much slower, probably because corrSelected isn't evidence
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) ^ corrSelected(c) ^ corrSelected(d) ^ sameObj(s, t) => srcsSameObj(c, d)." << endl;
	outfile << "corrHasSrc(c, s) ^ corrHasSrc(d, t) ^ corrSelected(c) ^ corrSelected(d) ^ srcsSameObj(c, d) => sameObj(s, t)." << endl;
	*/
#endif
#endif

	outfile << "sameObj(s,t) => sameObj(t,s)." << endl;
//		outfile << "sameScene(s,t) ^ sameScene(s,u) ^ sameObj(s,t) ^ sameObj(t,u) => sameObj(s,u)." << endl; //the precondition keeps the explosion of ground clauses down
	/*
	 * if not using MAP inference, this makes it very very slow (not quite as bad if use -saTemperature 10, -saRatio 0)
	 *
	 * if doing MAP, this actually uses less memory w/o any precondition than with sameScene(s,t) ^ sameScene(s,u) conjuncted in
	 */
	outfile << "sameObj(s,t) ^ sameObj(t,u) => sameObj(s,u)." << endl;

	outfile.close();

	/*
	write evidence
	*/
	outfile.open((outdir / "evidence.db").string());
	ASSERT_ALWAYS(outfile);

	/*
	 * per-scene evidence
	 */
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
			std::move(loadForegroundSegments(*scenes[i], outdir / mlnSegFilebase));

		/*
		 * sameScene
		 */
		for(unsigned int j = 0; j < numSegments1; j++)
			for(unsigned int k = 0; k < numSegments1; k++)
				if(j != k)
					outfile << "sameScene(" << compIDStr(i, j) << "," << compIDStr(i, k) << ")" << endl;

		/********************************************************************************************************
		 * evidence from neighboring-segment similarity measures
		 */

		unordered_map<unsigned int, unordered_set<unsigned int>> segNbrsWithSimilarNormals, segNbrsWithConcaveJunction, segNbrsWithConvexJunction;
		std::tie(segNbrsWithSimilarNormals, segNbrsWithConcaveJunction, segNbrsWithConvexJunction) =
			std::move(getNeighboringSegmentSimilarities(*scenes[i], numSegments1, surfel2segIndex1, segmentCentroids1, segNbrs1, outdir / "segNbrSims"));

		for(auto j = segNbrs1.begin(); j != segNbrs1.end(); j++)
		{
			for(auto k = (*j).second.begin(); k != (*j).second.end(); k++)
			{
				if(segNbrsWithSimilarNormals[(*j).first].find(*k) != segNbrsWithSimilarNormals[(*j).first].end())
					outfile << "segNbrsAndSimilarNormals(" << compIDStr(i, (*j).first) << "," << compIDStr(i, *k) << ")" << endl;
				else
					outfile << "segNbrs(" << compIDStr(i, (*j).first) << "," << compIDStr(i, *k) << ")" << endl;

				if(segNbrsWithConcaveJunction[(*j).first].find(*k) != segNbrsWithConcaveJunction[(*j).first].end())
				{
					outfile << "segNbrsConcave(" << compIDStr(i, (*j).first) << "," << compIDStr(i, *k) << ")" << endl;
				}
				else if(segNbrsWithConvexJunction[(*j).first].find(*k) != segNbrsWithConvexJunction[(*j).first].end())
				{
					outfile << "segNbrsConvex(" << compIDStr(i, (*j).first) << "," << compIDStr(i, *k) << ")" << endl;
				}
			}
		}

		/********************************************************************************************************
		 * hack: components not close in space are different objs
		 * TODO remove
		 */
		for(unsigned int j = 0; j < numSegments1 - 1; j++)
			for(unsigned int k = j + 1; k < numSegments1; k++)
				if(minCompSpatialDist1[j][k] > sqr(.55)) //dino is 52 cm long
					outfile << "!sameObj(" << compIDStr(i, j) << "," << compIDStr(i, k) << ")" << endl;

		/********************************************************************************************************
		 * like same xforms
		 */
		for(auto j = likeLikes[i].begin(); j != likeLikes[i].end(); j++)
			for(auto k = (*j).second.begin(); k != (*j).second.end(); k++)
				outfile << "likeLike(" << compIDStr(i, (*j).first) << "," << compIDStr(i, *k) << ")" << endl;
		for(auto j = likeDislikes[i].begin(); j != likeDislikes[i].end(); j++)
			for(auto k = (*j).second.begin(); k != (*j).second.end(); k++)
				outfile << "likeDislike(" << compIDStr(i, (*j).first) << "," << compIDStr(i, *k) << ")" << endl;
	}

#ifdef USE_CORRS
	/*
	 * interscene correspondence
	 */
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
			std::move(loadForegroundSegments(scenes[i], outdir / mlnSegFilebase));

		for(unsigned int j = 0; j < numScenes; j++)
			if(j != i)
			{
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
					std::move(loadForegroundSegments(scenes[j], outdir / mlnSegFilebase));

				for(auto k = allLegalSegCorrs[i][j].begin(); k != allLegalSegCorrs[i][j].end(); k++) //for each corr from scene i
					for(auto l = (*k).second.begin(); l != (*k).second.end(); l++)
					{
						outfile << "corrHasSrc(" << corrIDStr(i, (*k).first, j, *l) << "," << compIDStr(i, (*k).first) << ")" << endl;
						outfile << "corrHasDest(" << corrIDStr(i, (*k).first, j, *l) << "," << compIDStr(j, *l) << ")" << endl;

						for(auto m = (*k).second.begin(); m != (*k).second.end(); m++)
							if(*m != *l/* different corrs */)
								outfile << "corrsSameSrc(" << corrIDStr(i, (*k).first, j, *l) << "," << corrIDStr(i, (*k).first, j, *m) << ")" << endl;

						for(auto m = allLegalSegCorrs[i][j].begin(); m != allLegalSegCorrs[i][j].end(); m++) //for each corr from scene i
							for(auto n = (*m).second.begin(); n != (*m).second.end(); n++)
							{
								if((*k).first != (*m).first/* different corrs */ && *n == *l)
									outfile << "corrsSameDest(" << corrIDStr(i, (*k).first, j, *l) << "," << corrIDStr(i, (*m).first, j, *n) << ")" << endl;

								if((*m).first != (*k).first || *n != *l)/* different corrs */
									outfile << "corrsSameScenes(" << corrIDStr(i, (*k).first, j, *l) << "," << corrIDStr(i, (*m).first, j, *n) << ")" << endl;
							}
					}

//					ofstream outfile3((outdir / ("ddists-" + scenes[i].sceneName + "-" + scenes[j].sceneName + ".mat")).string());
//					ASSERT_ALWAYS(outfile3);
				for(unsigned int k = 0; k < numSegments1; k++)
					for(unsigned int m = 0; m < numSegments1; m++)
					{
						const float scene1intersegDist = (segmentCentroids1[k] - segmentCentroids1[m]).norm();
						for(auto l = allLegalSegCorrs[i][j][k].begin(); l != allLegalSegCorrs[i][j][k].end(); l++)
							for(auto n = allLegalSegCorrs[i][j][m].begin(); n != allLegalSegCorrs[i][j][m].end(); n++)
								if(m != k || *l != *n)
								{
									const float scene2intersegDist = (segmentCentroids2[*l] - segmentCentroids2[*n]).norm();
									const float ddist = fabs(scene2intersegDist - scene1intersegDist);
//										outfile3 << ddist << ' ';
									if(ddist < .03/*.8 * avgSegDiameter*//* TODO parameterize */) //TODO move the numerical bit above this section
										;//outfile << "corrsDistCompatible(" << corrIDStr(i, k, j, *l) << "," << corrIDStr(i, m, j, *n) << ")" << endl;
									else
										outfile << "corrsDistIncompatible(" << corrIDStr(i, k, j, *l) << "," << corrIDStr(i, m, j, *n) << ")" << endl;
								}
					}
			}
	}
#endif

	outfile.close();
	t.stop("write alchemy inputs");

	/*
	run alchemy

	does -lazyLowState help w/ speed? -- no (still true 20100902)

	don't need -lazy if using -memLimit? -- do need it

	don't use memLimit; Marc says it's deprecated due to being unreliable
	*/
	string queryFormulae = "sameObj";
#ifdef USE_CORRS
	queryFormulae += ",corrSelected";
#endif
	ostringstream outstr;
	outstr << "/home/eherbst/software/alchemy/bin/infer -i " << (outdir / "domain.mln") << " -e " << (outdir / "evidence.db") << " -r " << (outdir / "alchemy.out") << " -q " << queryFormulae << " -lazy";
	/*
	 * if using transitivity, add: -saTemperature 10 -saRatio 0
	 * (instead of tens of minutes w/ transitivity or 5-10 sec w/o, MC-SAT will take 1.5 min)
	 *
	 * for MAP inference, add -m -mwsMaxSteps N (default N is 1000000)
	 */
	const bool inferenceIsMAP = true;
	const bool usingTransitivity = true;
	if(inferenceIsMAP) outstr << " -m -mwsMaxSteps 10000";
	else
	{
		outstr << " -maxSteps 1000";
		if(usingTransitivity) outstr << " -saTemperature 10 -saRatio 0";
	}
	cout << "to run: '" << outstr.str() << "'" << endl;
}
