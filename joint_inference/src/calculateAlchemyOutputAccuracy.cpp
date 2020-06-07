/*
 * calculateAlchemyOutputAccuracy: calculate accuracy measure(s) for an alchemy output file wrt a "truth" labeling for a scene pair
 *
 * Evan Herbst
 * 9 / 4 / 10
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
#include "rgbd_util/timer.h"
#include "rgbd_util/mathUtils.h"
#include "rgbd_util/ioUtils.h"
#include "rgbd_util/ros_utility.h"
#include "evh_util/rosUtils.h" //hash<ros::Time>
#include "evh_util/fsUtils.h"
#include "evh_util/strongType.h"
#include "evh_util/serialization/unordered_map.h"
#include "evh_util/serialization/unordered_set.h"
#include "evh_util/visualizationUtils.h"
#include "evh_util/pointsNLines.h"
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

const unsigned int firstOutlierID = 11; //anything at least this large will be taken to mean the segment isn't true foreground and gets its own obj id
bool truthSame(const unsigned int trueLabel1, const unsigned int trueLabel2)
{
	return trueLabel1 < firstOutlierID && trueLabel2 < firstOutlierID && trueLabel1 == trueLabel2;
}

/*
 * arguments: mln outfile, # scenes, scene #s to look for, true-label filepaths
 */
int main(int argc, char* argv[])
{
	ASSERT_ALWAYS(argc >= 2);

	unsigned int _ = 1;
	const fs::path mlnOutpath(argv[_++]);
	const unsigned int numScenes = lexical_cast<unsigned int>(argv[_++]);
	ASSERT_ALWAYS(numScenes == 2); //or update this file
	const vector<unsigned int> sceneIndices = {lexical_cast<unsigned int>(argv[_++]), lexical_cast<unsigned int>(argv[_++])};
	unordered_map<unsigned int, unsigned int> indexMap;
	for(unsigned int i = 0; i < sceneIndices.size(); i++) indexMap[sceneIndices[i]] = i;
	vector<fs::path> truthFilepaths(numScenes);
	for(unsigned int i = 0; i < numScenes; i++) truthFilepaths[i] = argv[_++];

	/*
	 * read ground truth
	 */
	vector<vector<unsigned int>> trueCompIDs(numScenes); //scene -> seg -> component
	for(unsigned int i = 0; i < numScenes; i++)
	{
		ifstream infile(truthFilepaths[i].string());
		ASSERT_ALWAYS(infile);
		unsigned int segID, compID;
		while(infile >> segID >> compID)
		{
			ASSERT_ALWAYS(segID == trueCompIDs[i].size());
			trueCompIDs[i].push_back(compID);
		}
	}

	vector<unsigned int> numFGComponents(numScenes);
	for(unsigned int i = 0; i < numScenes; i++)
	{
		numFGComponents[i] = trueCompIDs[i].size();
		cout << "scene " << i << ": " << numFGComponents[i] << " segments" << endl;
	}

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
	ifstream infile(mlnOutpath.string());
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
			cout << scene1 << ' ' << comp1 << ' ' << scene2 << ' ' << comp2 << endl;
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

	float accuracy = 0;
	unsigned int count = 0;
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < numFGComponents[i] - 1; j++)
			for(unsigned int k = j + 1; k < numFGComponents[i]; k++)
			{
				const bool shouldBeSame = truthSame(trueCompIDs[i][j], trueCompIDs[i][k]),
					areSame = (representative2index[sets.find_set(std::make_pair(indexMap[i], j))] == representative2index[sets.find_set(std::make_pair(indexMap[i], k))]);
				if(shouldBeSame == areSame) accuracy++;
				count++;
			}
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < numScenes; j++)
			if(j != i)
				for(unsigned int k = 0; k < numFGComponents[i]; k++)
					for(unsigned int l = 0; l < numFGComponents[j]; l++)
					{
						const bool shouldBeSame = truthSame(trueCompIDs[i][k], trueCompIDs[j][l]),
							areSame = (representative2index[sets.find_set(std::make_pair(indexMap[i], k))] == representative2index[sets.find_set(std::make_pair(indexMap[j], l))]);
						if(shouldBeSame == areSame) accuracy++;
						count++;
					}
	accuracy /= count;
	cout << "accuracy on seg pairs: " << accuracy << endl;

	return 0;
}
