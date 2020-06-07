/*
 * mlntests: run an mln on simple synthetic datasets to see where it breaks
 *
 * Evan Herbst
 * 12 / 15 / 10
 */

#include <cassert>
#include <cstdlib> //system()
#include <cstdio> //getchar()
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
using std::vector;
using std::set;
using std::pair;
using std::string;
using std::ifstream;
using std::ofstream;
using std::ostringstream;
using std::cout;
using std::endl;
using boost::lexical_cast;
namespace fs = boost::filesystem;

const unsigned int squareSize = 4;

void runTest(const bool learning, const unsigned int numObjs, vector<vector<set<int>>>& segNbrs, vector<vector<vector<set<int>>>>& segCorrs, vector<vector<vector<vector<set<pair<int, int>>>>>>& badCorrPairs, const fs::path& mlnOutdir)
{
	const unsigned int numScenes = segNbrs.size();
	fs::create_directories(mlnOutdir);

	ofstream outfile;

	/*
	write mln file
		define types and constants
	*/
#define compIDStr(scene, seg) ("C" + lexical_cast<string>(scene) + "_" + lexical_cast<string>(seg))
#define objIDStr(oid) ("O" + lexical_cast<string>(oid))
	outfile.open((mlnOutdir / "domain.mln").string());
	ASSERT_ALWAYS(outfile);

	outfile << "comp = { ";
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < segNbrs[i].size(); j++)
			outfile << compIDStr(i, j) << ' ';
	outfile << "}" << endl;

	outfile << "oid = { ";
	for(unsigned int i = 0; i < numObjs; i++)
		outfile << objIDStr(i) << ' ';
	outfile << "}" << endl;

	outfile << endl;

	/*
	write mln file
		define predicates
		define formulae
	*/

	outfile << "sameScene(comp,comp)" << endl;
	outfile << "segNbrs(comp,comp)" << endl; //for all pairs of close segments
	outfile << "corrProposed(comp,comp)" << endl;
	outfile << "corrsIncompatible(comp,comp,comp,comp)" << endl;
	//outfile << "obj(comp,oid!)" << endl;
	outfile << "sameObj(comp,comp)" << endl;
	outfile << "corr(comp,comp!)" << endl;
	outfile << endl;

	if(learning)
	{
		outfile << 1 << " segNbrs(s,t) => sameObj(s,t)" << endl;
		outfile << "!corrProposed(s,t) => !corr(s,t)." << endl;
		outfile << .2 << " corrProposed(s,t) => corr(s,t)" << endl;
		outfile << -.1 << " corr(s,t)" << endl;
		outfile << "corr(s,t) => corr(t,s)." << endl;
		outfile << "corr(s,t) => sameObj(s,t)." << endl;
		outfile << "corrsIncompatible(s,t,u,v) => !(sameObj(s,t) ^ corr(s,t) ^ corr(u,v))." << endl;
	}
	else //inference
	{
#if 1
#if 1 //works on 4x4 w/ 20% outliers, 20110219; TODO figure out how to get objness too
		//outfile << "sameObj(s,s)." << endl;
		outfile << 10 << " segNbrs(s,t) => sameObj(s,t)" << endl;
		outfile << 2 << "corrsIncompatible(s,t,u,v) => !(sameObj(s,t) ^ corr(s,t) ^ corr(u,v))" << endl;
		outfile << 1 << " corrProposed(s,t) => corr(s,t)" << endl;
		//outfile << 1 << " !sameObj(s,t)" << endl;
		outfile << "corr(s,t) => corr(t,s)." << endl;
		outfile << "corr(s,t) => sameObj(s,t)." << endl;
		outfile << "!corrProposed(s,t) => !corr(s,t)." << endl;
#else //doesn't quite work
		outfile << 10 << " segNbrs(s,t) => (obj(s,o) <=> obj(t,o))" << endl;
		outfile << 2 << "corrsIncompatible(s,t,u,v) => !(obj(s,o) ^ obj(u,o) ^ corr(s,t) ^ corr(u,v))" << endl;
		outfile << 1 << " corrProposed(s,t) => corr(s,t)" << endl;
		outfile << "corr(s,t) => corr(t,s)." << endl;
		outfile << "corr(s,t) ^ obj(s,o) => obj(t,o)." << endl;
		outfile << "!corrProposed(s,t) => !corr(s,t)." << endl;
#endif
#elif 1 //works on 4x4 w/ 20% outliers, 20110219
		outfile << 10 << " segNbrs(s,t) => (obj(s,o) <=> obj(t,o))" << endl;
		outfile << 2 << " corrsIncompatible(s,t,u,v) => !(corr(s,t) ^ corr(u,v))" << endl;
		outfile << 1 << " corrProposed(s,t) => corr(s,t)" << endl;
		outfile << "corr(s,t) => corr(t,s)." << endl;
		outfile << "corr(s,t) ^ obj(s,o) => obj(t,o)." << endl;
		outfile << "!corrProposed(s,t) => !corr(s,t)." << endl;
#elif 1 //works on 4x4 w/ 20% outliers, 20110218
		outfile << 10 << "segNbrs(s,t) => (obj(s,o) <=> obj(t,o))" << endl;
		outfile << 2 << " segNbrs(s,t) ^ corr(s,s1) ^ corr(t,t1) => segNbrs(s1,t1)" << endl;
		outfile << 1 << " corrProposed(s,t) => corr(s,t)" << endl;
		outfile << 10 << "corr(s,t) => corr(t,s)" << endl;
		outfile << 10 << "corr(s,t) ^ obj(s,o) => obj(t,o)" << endl;
		outfile << "sameScene(c1,c2) => !corr(c1,c2)." << endl;
#endif
	}

	outfile.close();

	/*
	write evidence
	*/
	outfile.open((mlnOutdir / "evidence.db").string());
	ASSERT_ALWAYS(outfile);
#if 1
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < segNbrs[i].size(); j++)
			for(unsigned int k = 0; k < segNbrs[i].size(); k++)
				if(j != k)
					outfile << "sameScene(" << compIDStr(i, j) << "," << compIDStr(i, k) << ")" << endl;
#endif
#if 1
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < segNbrs[i].size(); j++)
			for(auto k = segNbrs[i][j].begin(); k != segNbrs[i][j].end(); k++)
				outfile << "segNbrs(" << compIDStr(i, j) << "," << compIDStr(i, *k) << ")" << endl;
#endif
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < numScenes; j++)
			if(j != i)
				for(unsigned int l = 0; l < segNbrs[i].size(); l++)
					for(auto k = segCorrs[i][j][l].begin(); k != segCorrs[i][j][l].end(); k++)
						outfile << "corrProposed(" << compIDStr(i, l) << "," << compIDStr(j, *k) << ")" << endl;
#if 1
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < numScenes; j++)
			if(j != i)
				for(unsigned int l = 0; l < segNbrs[i].size(); l++)
					for(unsigned int m = 0; m < segNbrs[i].size(); m++)
						if(m != l)
							for(auto n = badCorrPairs[i][j][l][m].begin(); n != badCorrPairs[i][j][l][m].end(); n++)
								outfile << "corrsIncompatible(" << compIDStr(i, l) << "," << compIDStr(j, (*n).first) << "," << compIDStr(i, m) << "," << compIDStr(j, (*n).second) << ")" << endl;
#endif
if(learning)
{
	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int j = 0; j < numScenes; j++)
			if(j != i)
				for(unsigned int l = 0; l < segNbrs[i].size(); l++)
					outfile << "corr(" << compIDStr(i, l) << "," << compIDStr(j, l) << ")" << endl;

	for(unsigned int i = 0; i < numScenes; i++)
		for(unsigned int l = 0; l < segNbrs[i].size(); l++)
			outfile << "obj(" << compIDStr(i, l) << "," << objIDStr((l < squareSize*squareSize) ? 0 : 1) << ")" << endl; //TODO hack for fixed square size
}

	outfile.close();

	if(learning)
	{
		const fs::path mlnOutpath = mlnOutdir / "learning.out";
		string queryFormulae = "sameObj,corr";//"obj,corr";
		ostringstream outstr;
		outstr << "/home/eherbst/software/alchemy/bin/learnwts -g -i " << (mlnOutdir / "domain.mln") << " -t " << (mlnOutdir / "evidence.db") << " -o " << mlnOutpath
				<< " -ne " << queryFormulae;

		system(outstr.str().c_str());
	}
	else //inference
	{
	/*
	run alchemy

	does -lazyLowState help w/ speed? -- no (still true 20100902)

	don't need -lazy if using -memLimit? -- do need it

	don't use memLimit; Marc says it's deprecated due to being unreliable
	*/
	const fs::path mlnOutpath = mlnOutdir / "alchemy.out";
	string queryFormulae = "sameObj,corr";//"obj,corr";
	ostringstream outstr;
	outstr << "/home/eherbst/software/alchemy/bin/infer -i " << (mlnOutdir / "domain.mln") << " -e " << (mlnOutdir / "evidence.db") << " -r " << mlnOutpath << " -q " << queryFormulae << " -lazy";
	/*
	 * for MAP inference, add -m -mwsMaxSteps N (default N is 100000)
	 */
	const bool inferenceIsMAP = true;
	if(inferenceIsMAP) outstr << " -m -mwsMaxSteps 3000000";
	else
	{
		//outstr << " -maxSteps 1000";
		outstr << " -ms -maxSteps 1500";// -mwsMaxSteps 3000000"; //default maxSteps 1000
//		outstr << " -saRatio 1"; //default 0; integer; TODO what is it?
	}

//	outstr << " >" << (mlnOutdir / "alchemy.cout") << " 2>" << (mlnOutdir / "alchemy.cerr");
	system(outstr.str().c_str());
#if 0
	/*
	read results
	*/
	vector<vector<int>> resultObjs(numScenes); //scene -> seg -> obj id
	vector<vector<vector<int>>> resultCorrs(numScenes); //scene1 -> scene2 -> sc1seg -> sc2seg
	for(unsigned int i = 0; i < numScenes; i++)
	{
		resultObjs[i].resize(segNbrs[i].size(), -1);
		resultCorrs[i].resize(numScenes);
		for(unsigned int j = 0; j < numScenes; j++)
			resultCorrs[i][j].resize(segNbrs[i].size(), -1);
	}
	boost::regex objRex("obj\\(\\w+(\\d+)_(\\d+),\\w+(\\d+)\\)"), corrRex("corr\\(\\w+(\\d+)_(\\d+),\\w+(\\d+)_(\\d+)\\)");
	ifstream infile(mlnOutpath.string());
	ASSERT_ALWAYS(infile);
	string predicate;
	while(infile >> predicate)
	{
		boost::smatch match;
		if(boost::regex_match(predicate, match, objRex))
		{
			const unsigned int scene = lexical_cast<unsigned int>(match[1]), comp = lexical_cast<unsigned int>(match[2]),
					obj = lexical_cast<unsigned int>(match[3]);
			resultObjs[scene][comp] = obj;
			cout << "sc" << scene << "seg" << comp << " obj: " << obj << endl;
		}
		else if(boost::regex_match(predicate, match, corrRex))
		{
			const unsigned int scene1 = lexical_cast<unsigned int>(match[1]), comp1 = lexical_cast<unsigned int>(match[2]),
				scene2 = lexical_cast<unsigned int>(match[3]), comp2 = lexical_cast<unsigned int>(match[4]);
			resultCorrs[scene1][scene2][comp1] = comp2;
			cout << "match: sc" << scene1 << "seg" << comp1 << " sc" << scene2 << "seg" << comp2 << endl;
		}
		else
		{
			cout << "no match: '" << predicate << "'" << endl;
			getchar();
		}
	}
	infile.close();

	/*
	 * result accuracy
	 */
//	double accuracy = 0;
//	for(unsigned int i = 0; i < numScenes; i++)
#endif
	}
}

/*****************************************************************************************************************/

/*
 * each (i,j) seg in each square has only element (i,j) of the other scene's corresponding square as a possible corr
 */
void createTwoSeparateSquaresDataset(unsigned int& numObjs, vector<vector<set<int>>>& segNbrs, vector<vector<vector<set<int>>>>& segCorrs, vector<vector<vector<vector<set<pair<int, int>>>>>>& badCorrPairs)
{
	numObjs = 2;

	const unsigned int n = squareSize; //square size
	for(unsigned int i = 0; i < 2; i++)
	{
		segNbrs[i].resize(2 * n * n);
		segCorrs[i][!i].resize(2 * n * n);
		badCorrPairs[i][!i].resize(2 * n * n);
		for(unsigned int m = 0; m < 2; m++)
			for(unsigned int k = 0; k < n; k++)
				for(unsigned int l = 0; l < n; l++)
				{
#define SEG(sc, sq, r, c) (/*(sc) * 2 * n * n + */(sq) * n * n + (r) * n + (c))
#define s SEG(i, m, k, l)
					badCorrPairs[i][!i][s].resize(2 * n * n);

					if(k > 0) segNbrs[i][s].insert(SEG(i, m, k - 1, l));
					if(k < n - 1) segNbrs[i][s].insert(SEG(i, m, k + 1, l));
					if(l > 0) segNbrs[i][s].insert(SEG(i, m, k, l - 1));
					if(l < n - 1) segNbrs[i][s].insert(SEG(i, m, k, l + 1));
					segCorrs[i][!i][s].insert(SEG(!i, m, k, l));
#undef s
#undef SEG
				}
	}
}

/*
 * each (i, j) seg in square k has element (i,j) in each square in the other scene as a possible corr;
 * include incompatible-corrs list
 */
void createTwoConfoundedSquaresDataset(unsigned int& numObjs, vector<vector<set<int>>>& segNbrs, vector<vector<vector<set<int>>>>& segCorrs, vector<vector<vector<vector<set<pair<int, int>>>>>>& badCorrPairs)
{
	srand((unsigned int)time(NULL));
	const double pctCrossCorrs = .2;

	numObjs = 2;

	const unsigned int n = squareSize; //square size
	for(unsigned int i = 0; i < 2; i++)
	{
		segNbrs[i].resize(2 * n * n);
		segCorrs[i][!i].resize(2 * n * n);
		badCorrPairs[i][!i].resize(2 * n * n);
	}
	for(unsigned int i = 0; i < 2; i++)
	{
		for(unsigned int m = 0; m < 2; m++)
			for(unsigned int k = 0; k < n; k++)
				for(unsigned int l = 0; l < n; l++)
				{
#define SEG(sc, sq, r, c) ((sq) * n * n + (r) * n + (c))
#define s SEG(i, m, k, l)
					badCorrPairs[i][!i][s].resize(2 * n * n);

					if(k > 0) segNbrs[i][s].insert(SEG(i, m, k - 1, l));
					if(k < n - 1) segNbrs[i][s].insert(SEG(i, m, k + 1, l));
					if(l > 0) segNbrs[i][s].insert(SEG(i, m, k, l - 1));
					if(l < n - 1) segNbrs[i][s].insert(SEG(i, m, k, l + 1));
					//corrs must be proposed in fwd-bkwd pairs if we're going to have the fwd-bkwd constraint in the mln
					segCorrs[i][!i][s].insert(SEG(!i, m, k, l));
					segCorrs[!i][i][s].insert(SEG(i, m, k, l));
					if(rand() % 100 < pctCrossCorrs * 100)
					{
						segCorrs[i][!i][s].insert(SEG(!i, !m, k, l));
						segCorrs[!i][i][SEG(!i, !m, k, l)].insert(SEG(i, m, k, l));
					}
#undef s
#undef SEG
				}

		for(unsigned int m = 0; m < 2; m++) //for each scene1 seg
			for(unsigned int k = 0; k < n; k++)
				for(unsigned int l = 0; l < n; l++)
				{
#define SEG(sc, sq, r, c) ((sq) * n * n + (r) * n + (c))
#define s SEG(i, m, k, l)
#define ROW(s) (((s) / n) % n)
#define COL(s) ((s) % n)
					for(auto q = segNbrs[i][s].begin(); q != segNbrs[i][s].end(); q++) //for each scene1 nbr
						for(unsigned int o = 0; o < 2; o++) //for each square in scene2
							for(unsigned int p = 0; p != 2; p++) //for each other scene2 square
								if(p != o)
									badCorrPairs[i][!i][s][*q].insert(std::make_pair(SEG(!i, o, k, l), SEG(!i, p, ROW(*q), COL(*q))));
#undef s
#undef SEG
#undef ROW
#undef COL
				}
	}
}

/*****************************************************************************************************************/

int main(int argc, char* argv[])
{
	ASSERT_ALWAYS(argc == 2);
	const fs::path outdir(argv[1]);

	const unsigned int numScenes = 2;
	unsigned int numObjs;
	vector<vector<set<int>>> segNbrs(numScenes); //scene -> seg 1 -> nbr segs
	vector<vector<vector<set<int>>>> segCorrs(numScenes); //scene1 -> scene2 -> scene-1 seg -> scene-2 segs
	vector<vector<vector<vector<set<pair<int, int>>>>>> badCorrPairs(numScenes); //scene1 -> scene2 -> sc1seg1 -> sc1seg2 -> (sc2seg1, sc2seg2)
	for(unsigned int i = 0; i < numScenes; i++)
	{
		segCorrs[i].resize(numScenes);
		badCorrPairs[i].resize(numScenes);
	}

#if 0
	createTwoSeparateSquaresDataset(numObjs, segNbrs, segCorrs, badCorrPairs);
#elif 1
	createTwoConfoundedSquaresDataset(numObjs, segNbrs, segCorrs, badCorrPairs);
#endif

	runTest(false/* learning = true, inference = false */, numObjs, segNbrs, segCorrs, badCorrPairs, outdir);

	return 0;
}
