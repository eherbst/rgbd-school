/*
 * parseAlchemyOutputClusterSizes: to debug getting alchemy to give me anything nontrivial
 *
 * Evan Herbst
 * 8 / 17 / 10
 */

#include <cassert>
#include <vector>
#include <unordered_set>
#include <string>
#include <iostream>
#include <fstream>
#include <utility>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/unordered_map.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/regex.hpp>
#include "rgbd_util/mathUtils.h" //hash<pair<>>
using std::vector;
using std::unordered_set;
using std::string;
using std::ifstream;
using std::pair;
using std::cout;
using std::endl;
using boost::lexical_cast;
namespace fs = boost::filesystem;

/*
 * arguments: alchemy outfilepath, whether inference was MAP
 */
int main(int argc, char* argv[])
{
	ASSERT_ALWAYS(argc == 3);
	const fs::path infilepath(argv[1]);
	const bool mapInference = lexical_cast<bool>(argv[2]);

	typedef boost::associative_property_map<boost::unordered_map<pair<unsigned int, unsigned int>, unsigned int> > rankMapT;
	typedef boost::associative_property_map<boost::unordered_map<pair<unsigned int, unsigned int>, pair<unsigned int, unsigned int>> > parentMapT;
	boost::unordered_map<pair<unsigned int, unsigned int>, unsigned int> rankMap;
	boost::unordered_map<pair<unsigned int, unsigned int>, pair<unsigned int, unsigned int>> parentMap;
	boost::disjoint_sets<rankMapT, parentMapT> sets(boost::make_assoc_property_map(rankMap), boost::make_assoc_property_map(parentMap));
	unordered_set<pair<unsigned int, unsigned int>> allPairs;

	ifstream infile;
	infile.open(infilepath.string().c_str());
	ASSERT_ALWAYS(infile);
	boost::regex sameRex("sameObj\\(\\w+(\\d+)_(\\d+),\\w+(\\d+)_(\\d+)\\)");
	string predicate;
	if(mapInference)
	{
		while(infile >> predicate)
		{
			boost::smatch match;
			if(boost::regex_match(predicate, match, sameRex))
			{
				const unsigned int scene1 = lexical_cast<unsigned int>(match[1]), comp1 = lexical_cast<unsigned int>(match[2]),
					scene2 = lexical_cast<unsigned int>(match[3]), comp2 = lexical_cast<unsigned int>(match[4]);
				if(allPairs.find(std::make_pair(scene1, comp1)) == allPairs.end())
				{
					allPairs.insert(std::make_pair(scene1, comp1));
					sets.make_set(std::make_pair(scene1, comp1));
				}
				if(allPairs.find(std::make_pair(scene2, comp2)) == allPairs.end())
				{
					allPairs.insert(std::make_pair(scene2, comp2));
					sets.make_set(std::make_pair(scene2, comp2));
				}
				sets.union_set(std::make_pair(scene1, comp1), std::make_pair(scene2, comp2));
			}
			else cout << "didn't match" << endl;
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
				if(allPairs.find(std::make_pair(scene1, comp1)) == allPairs.end())
				{
					allPairs.insert(std::make_pair(scene1, comp1));
					sets.make_set(std::make_pair(scene1, comp1));
				}
				if(allPairs.find(std::make_pair(scene2, comp2)) == allPairs.end())
				{
					allPairs.insert(std::make_pair(scene2, comp2));
					sets.make_set(std::make_pair(scene2, comp2));
				}
				if(prob > .5) sets.union_set(std::make_pair(scene1, comp1), std::make_pair(scene2, comp2));
			}
			else cout << "didn't match" << endl;
		}
	}
	infile.close();
	cout << "got " << allPairs.size() << " pairs" << endl;

	/*
	 * get the list of representatives and map them to [0 .. n)
	 */
	boost::unordered_map<pair<unsigned int, unsigned int>, unsigned int> representative2index; //map set IDs into the integer range [0 .. n) for some n
	for(auto i = allPairs.begin(); i != allPairs.end(); i++)
		{
			const pair<unsigned int, unsigned int> rep = sets.find_set(*i);
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
	for(auto i = allPairs.begin(); i != allPairs.end(); i++)
			cluster2comps[representative2index[sets.find_set(*i)]].insert(*i);
	cout << "cluster sizes > 1:";
	for(unsigned int j = 0; j < cluster2comps.size(); j++)
		if(cluster2comps[j].size() > 1)
			cout << ' ' << cluster2comps[j].size() << endl;
	cout << endl;

	return 0;
}
