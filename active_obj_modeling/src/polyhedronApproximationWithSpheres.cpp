/*
 * polyhedronApproximationWithSpheres: for such things as carving free space into a voxel map
 *
 * Evan Herbst
 * 11 / 26 / 13
 */

#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <ros/package.h>
//spheretree is a library for approximating polyhedra with overlapping spheres
#include "spheretree/Surface/Surface.h"
#include "spheretree/API/MSGrid.h"
#include "spheretree/API/SEConvex.h"
#include "spheretree/API/SESphPt.h"
#include "spheretree/API/VFAdaptive.h"
#include "spheretree/API/SSIsohedron.h"
#include "spheretree/API/SRMerge.h"
#include "spheretree/API/SRExpand.h"
#include "spheretree/API/SRBurst.h"
#include "spheretree/API/SRComposite.h"
#include "spheretree/API/SFWhite.h"
#include "spheretree/API/REMaxElim.h"
#include "spheretree/API/STGGeneric.h"
#include "spheretree/API/SOSimplex.h"
#include "spheretree/API/SOBalance.h"
#include "spheretree/VerifyModel.h"
#include "rgbd_util/assert.h"
#include "rgbd_util/mathUtils.h"
#include "active_obj_modeling/polyhedronApproximationWithSpheres.h"
using std::ifstream;
using std::ofstream;
using std::cout;
using std::endl;
namespace fs = boost::filesystem;

namespace rgbd
{

/*
 * mins and maxes are of an axis-aligned bbox
 *
 * objName is used for on-disk caching
 */
std::vector<sphereInfo> approximateAABBWithSpheres(const rgbd::eigen::Vector3f& mins, const rgbd::eigen::Vector3f& maxes, const std::string& objName)
{
	std::vector<sphereInfo> spheres; //to be read from disk or computed

	const fs::path sphereListFilepath = fs::path(ros::package::getPath("active_obj_modeling")) / "data" / (boost::format("bboxSphereList-%1%.dat") % objName).str();
	if(fs::exists(sphereListFilepath))
	{
		ifstream infile(sphereListFilepath.string());
		ASSERT_ALWAYS(infile);
		sphereInfo s;
		while(infile >> s.c.x() >> s.c.y() >> s.c.z() >> s.r) spheres.push_back(s);
	}
	else
	{
		/*
		 * create a triangulated mesh for the bbox
		 */
		std::vector<Point3D> vertices(8);
		std::vector<Surface::triverts> triangles(12);
		for(size_t q = 0; q < 8; q++)
		{
			Point3D v;
			v.x = mins.x() + ((q >> 2) & 1) * (maxes.x() - mins.x());
			v.y = mins.y() + ((q >> 1) & 1) * (maxes.y() - mins.y());
			v.z = mins.z() + ((q >> 0) & 1) * (maxes.z() - mins.z());
			vertices[q] = v;
		}
		triangles[0] = Surface::triverts(0, 1, 3);
		triangles[1] = Surface::triverts(0, 3, 2);
		triangles[2] = Surface::triverts(1, 5, 7);
		triangles[3] = Surface::triverts(1, 7, 3);
		triangles[4] = Surface::triverts(0, 4, 5);
		triangles[5] = Surface::triverts(0, 5, 1);
		triangles[6] = Surface::triverts(6, 3, 7);
		triangles[7] = Surface::triverts(6, 2, 3);
		triangles[8] = Surface::triverts(0, 6, 4);
		triangles[9] = Surface::triverts(0, 2, 6);
		triangles[10] = Surface::triverts(4, 7, 5);
		triangles[11] = Surface::triverts(4, 6, 7);

		/*
		 * use the spheretree library to approximate a polyhedron with spheres
		 */
		/*
			 options and their default values
		*/
		int testerLevels = -1;      //  number of levels for NON-CONVEX, -1 uses CONVEX tester
		int branch = 8;             //  branching factor of the sphere-tree
		int depth = 3;              //  depth of the sphere-tree
		int numCoverPts = 5000;     //  number of test points to put on surface for coverage
		int minCoverPts = 5;        //  minimum number of points per triangle for coverage
		int initSpheres = 500;      //  initial spheres in Voronoi
		float erFact = 2;           //  error reduction factor for adaptive Voronoi
		int spheresPerNode = 100;   //  minimum number of spheres per node
	//	bool eval = false;          //  do we evaluate the sphere-tree after construction
		bool useMerge = false;      //  do we include the MERGE algorithm
		bool useBurst = false;      //  do we include the BURST algorithm
		bool useExpand = false;     //  do we include the EXPAND algorithm
		enum Optimisers {NONE, SIMPLEX, BALANCE};
		int optimise = NONE;        //  which optimiser should we use
		float balExcess = 0.0;      //  % increase error allowed for BALANCE opt algorithm
		int maxOptLevel = -1;       //  maximum sphere-tree level to apply optimiser (0 does first set only)

		const uint64_t DO_ALL_BELOW = branch*3;
		const uint64_t MAX_ITERS_FOR_VORONOI = 1.50*spheresPerNode;
		if (!useMerge && !useBurst && !useExpand)
			useMerge = useExpand = TRUE;        //  default to combined algorithm

		Surface sur;
		sur.loadTriangleMesh(vertices, triangles);
		/*
			scale box
		*/
		float boxScale = sur.fitIntoBox(1000); //EVH: 1000 was in the original spheretree code -- ??
		/*
			make medial tester
		*/
		MedialTester mt;
		mt.setSurface(sur);
		mt.useLargeCover = true;
		/*
			set up evaluator
		*/
		SEConvex convEval;
		convEval.setTester(mt);
		SEBase *eval = &convEval;

		Array<Point3D> sphPts;
		SESphPt sphEval;
		if (testerLevels > 0){   //  <= 0 will use convex tester
		 SSIsohedron::generateSamples(&sphPts, testerLevels-1);
		 sphEval.setup(mt, sphPts);
		 eval = &sphEval;
		 printf("Using concave tester (%d)\n\n", sphPts.getSize());
		 }
		/*
			verify model
		*/
		ASSERT_ALWAYS(verifyModel(sur)); //model is unusable if this fails (basically it needs to be a watertight and bounded mesh)
		/*
			set up for the set of cover points
		*/
		Array<Surface::Point> coverPts;
		MSGrid::generateSamples(&coverPts, numCoverPts, sur, TRUE, minCoverPts);
		/*
		  Set up voronoi diagram
		*/
		Point3D pC;
		pC.x = (sur.pMax.x + sur.pMin.x)/2.0f;
		pC.y = (sur.pMax.y + sur.pMin.y)/2.0f;
		pC.z = (sur.pMax.z + sur.pMin.z)/2.0f;
		Voronoi3D vor;
		vor.initialise(pC, 1.5f * sur.pMin.distance(pC));
		/*
			set up adaptive Voronoi algorithm
		*/
		VFAdaptive adaptive;
		adaptive.mt = &mt;
		adaptive.eval = eval;
		/*
			set up FITTER
		*/
		SFWhite fitter;
		/*
			set up MERGE
		*/
		SRMerge merger;
		merger.sphereEval = eval;
		merger.sphereFitter = &fitter;
		merger.useBeneficial = true;
		merger.doAllBelow = DO_ALL_BELOW;
		merger.setup(&vor, &mt);
		merger.vorAdapt = &adaptive;
		merger.initSpheres = initSpheres;
		merger.errorDecreaseFactor = erFact;
		merger.minSpheresPerNode = spheresPerNode;
		merger.maxItersForVoronoi = MAX_ITERS_FOR_VORONOI;
		/*
			set up BURST
		*/
		SRBurst burster;
		burster.sphereEval = eval;
		burster.sphereFitter = &fitter;
		burster.useBeneficial = true;
		burster.doAllBelow = DO_ALL_BELOW;
		burster.setup(&vor, &mt);
		if (!useBurst){
		 //  set up adaptive algorithm
		 burster.vorAdapt = &adaptive;
		 burster.initSpheres = initSpheres;
		 burster.errorDecreaseFactor = erFact;
		 burster.minSpheresPerNode = spheresPerNode;
		 burster.maxItersForVoronoi = MAX_ITERS_FOR_VORONOI;
		 }
		else{
		 // leave adaptive algorithm out as merge will do it for us
		 burster.vorAdapt = NULL;
		 }
		/*
			set up EXPAND generator
		*/
		REMaxElim elimME;
		SRExpand expander;
		expander.redElim = &elimME;
		expander.setup(&vor, &mt);
		expander.errStep = 100;
		expander.useIterativeSelect = false;
		expander.relTol = 1E-5;
		if (!useMerge && !useBurst){
		 //  set up adaptive algorithm
		 expander.vorAdapt = &adaptive;
		 expander.initSpheres = initSpheres;
		 expander.errorDecreaseFactor = erFact;
		 expander.minSpheresPerNode = spheresPerNode;
		 expander.maxItersForVoronoi = MAX_ITERS_FOR_VORONOI;
		 }
		else{
		 // leave adaptive algorithm out as previous algs will do it for us
		 expander.vorAdapt = NULL;
		 }
		/*
		 set up the COMPOSITE algorithm
		*/
		SRComposite composite;
		composite.eval = eval;
		if (useMerge) composite.addReducer(&merger);
		if (useBurst) composite.addReducer(&burster);
		if (useExpand) composite.addReducer(&expander);
		/*
			set up simplex optimiser in case we want it
		*/
		SOSimplex simOpt;
		simOpt.sphereEval = eval;
		simOpt.sphereFitter = &fitter;
		/*
			set up balance optimiser to throw away spheres as long as
			increase in error is less than balExcess e.g. 0.05 is 1.05%
		*/
		SOBalance balOpt;
		balOpt.sphereEval = eval;
		balOpt.optimiser = &simOpt;
		balOpt.V = 0.0f;
		balOpt.A = 1;
		balOpt.B = balExcess;
		/*
			set up SphereTree constructor - using dynamic construction
		*/
		STGGeneric treegen;
		treegen.eval = eval;
		treegen.useRefit = true;
		treegen.setSamples(coverPts);
		treegen.reducer = &composite;
		treegen.maxOptLevel = maxOptLevel;
		if (optimise == SIMPLEX) treegen.optimiser = &simOpt;
		else if (optimise == BALANCE) treegen.optimiser = &balOpt;
		/*
			make sphere tree
		*/
		SphereTree tree;
		tree.setupTree(branch, depth+1);
		treegen.constructTree(&tree);
		/*
		 * extract spheres (TODO what is occupancy? does it matter for us?)
		 */
		const std::vector<float> sphereTreeSpec = tree.serializeSphereTree(1.0f/boxScale);
		const size_t numLevels = sphereTreeSpec[0], degree = sphereTreeSpec[1];
		cout << "sphere tree: " << numLevels << " levels, degree " << degree << endl;
		const size_t numNodes = (sphereTreeSpec.size() - 2) / 5;
		for(size_t i = numNodes - powi(degree, numLevels - 1); i < numNodes; i++) //read just the last level of spheres--each level is a refinement of the one above
		{
			sphereInfo s;
			s.c.x() = sphereTreeSpec[2 + i * 5 + 0];
			s.c.y() = sphereTreeSpec[2 + i * 5 + 1];
			s.c.z() = sphereTreeSpec[2 + i * 5 + 2];
			s.r = sphereTreeSpec[2 + i * 5 + 3];
			if(s.r > 1e-4) //< 0 is nonvalid, and some seem to have very small numbers like 8e-17; don't know why
				spheres.push_back(s);
		}

		/*
		 * add the bbox center because the sphere positions seem to be centered around zero but do have about the right extent
		 */
		const rgbd::eigen::Vector3f bboxCenter = (mins + maxes) / 2;
		for(sphereInfo& s : spheres) s.c += bboxCenter;

		/*
		 * write to disk
		 */
		fs::create_directories(sphereListFilepath.parent_path());
		ofstream outfile(sphereListFilepath.string());
		ASSERT_ALWAYS(outfile);
		for(const sphereInfo& s : spheres) outfile << s.c.transpose() << ' ' << s.r << endl;
	}

	return spheres;
}

} //namespace
