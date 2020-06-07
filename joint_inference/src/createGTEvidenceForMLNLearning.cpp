/*
 * createGTEvidenceForMLNLearning: take true segment labels and write evidence to be appended to the alchemy input for weight learning
 *
 * Evan Herbst
 * 9 / 7 / 10
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
 * arguments: mln evidence filepath, # scenes, scene #s to look for, true-label filepaths, outfilepath
 */
int main(int argc, char* argv[])
{
	ASSERT_ALWAYS(argc >= 2);

	unsigned int _ = 1;
	const fs::path evidenceFilepath(argv[_++]);
	const unsigned int numScenes = lexical_cast<unsigned int>(argv[_++]);
	ASSERT_ALWAYS(numScenes == 2); //or update this file
	const vector<unsigned int> sceneIndices = {lexical_cast<unsigned int>(argv[_++]), lexical_cast<unsigned int>(argv[_++])};
	unordered_map<unsigned int, unsigned int> indexMap;
	for(unsigned int i = 0; i < sceneIndices.size(); i++) indexMap[sceneIndices[i]] = i;
	vector<fs::path> truthFilepaths(numScenes);
	for(unsigned int i = 0; i < numScenes; i++) truthFilepaths[i] = argv[_++];
	const fs::path outfilepath(argv[_++]);

	/*
	 * read evidence
	 */
	vector<pair<unsigned int, unsigned int>> proposedCorrs; //(scene-1 seg, scene-2 seg)
	boost::regex rex("corrProposed\\(\\w+(\\d+)_(\\d+),\\w+(\\d+)_(\\d+)\\)");
	ifstream infile(evidenceFilepath.string());
	ASSERT_ALWAYS(infile);
	string predicate;
	while(infile >> predicate)
	{
		boost::smatch match;
		if(boost::regex_match(predicate, match, rex))
		{
			const unsigned int scene1 = indexMap[lexical_cast<unsigned int>(match[1])], comp1 = lexical_cast<unsigned int>(match[2]),
				scene2 = indexMap[lexical_cast<unsigned int>(match[3])], comp2 = lexical_cast<unsigned int>(match[4]);
//			cout << scene1 << ' ' << comp1 << ' ' << scene2 << ' ' << comp2 << endl;
			if(scene1 == 0) proposedCorrs.push_back(std::make_pair(comp1, comp2));
			else proposedCorrs.push_back(std::make_pair(comp2, comp1));
		}
	}
	infile.close();
	cout << "got " << proposedCorrs.size() << " corrs" << endl;

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
	 * write evidence
	 */
	ofstream outfile(outfilepath.string());
	ASSERT_ALWAYS(outfile);

#define compIDStr(scene, seg) ("C" + lexical_cast<string>(scene) + "_" + lexical_cast<string>(seg))

	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int k = 0; k < numFGComponents[i]; k++)
		{
			for(unsigned int l = 0; l < numFGComponents[i]; l++)
				if(l != k)
				{
					if(trueCompIDs[i][k] == trueCompIDs[i][l])
						outfile << "sameObj(" << compIDStr(i, k) << "," << compIDStr(i, l) << ")" << endl;
					else
						outfile << "!sameObj(" << compIDStr(i, k) << "," << compIDStr(i, l) << ")" << endl;
				}
			for(unsigned int j = 0; j < numScenes; j++)
				if(j != i)
				{
					for(unsigned int l = 0; l < numFGComponents[j]; l++)
						if(trueCompIDs[i][k] == trueCompIDs[j][l])
							outfile << "sameObj(" << compIDStr(i, k) << "," << compIDStr(j, l) << ")" << endl;
						else
							outfile << "!sameObj(" << compIDStr(i, k) << "," << compIDStr(j, l) << ")" << endl;
				}
		}

	for(auto i = proposedCorrs.begin(); i != proposedCorrs.end(); i++)
		if(trueCompIDs[0][(*i).first] != trueCompIDs[1][(*i).second])
			outfile << "!corrSelected(" << compIDStr(0, (*i).first) << "," << compIDStr(1, (*i).second) << ")" << endl;

#undef compIDStr

	outfile.close();

	return 0;
}
