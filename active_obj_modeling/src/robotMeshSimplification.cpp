/*
 * robotMeshSimplification: reduce # triangles and otherwise make meshes easier to use
 *
 * Evan Herbst
 * 12 / 4 / 13
 */

#define USE_MY_CODE
#ifndef USE_MY_CODE
#define USE_PROGMESH //other people's code: progmesh or cgal
#endif

#include <array>
#include <iostream>
#ifndef USE_MY_CODE
#ifndef USE_PROGMESH
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h> // Extended polyhedron items which include an id() field
#include <CGAL/Surface_mesh_simplification/HalfedgeGraph_Polyhedron_3.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h> // Simplification function
#include <CGAL/Surface_mesh_simplification/Edge_collapse_visitor_base.h> //callbacks during simplification
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h> // Stop-condition policy
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_and_length.h> // Non-default cost and placement policies
#else
#include "active_obj_modeling/progressiveMeshes/quadrics.h" //mesh simplification
#endif
#endif
#include "active_obj_modeling/robotMeshSimplification.h"
using std::cout;
using std::endl;

/*
 * create versions with a reduced number of triangles, whether approximate meshes or just a submesh, for faster collision checking and maybe other purposes
 *
 * return: meshes; which link each mesh comes from
 */
std::tuple<std::vector<triangulatedMesh>, std::vector<uint32_t>> createSimplifiedRobotMeshes(const OpenRAVE::RobotBasePtr& robot, const robotMeshSimplificationParams& params)
{
	std::tuple<std::vector<triangulatedMesh>, std::vector<uint32_t>> result;
	std::vector<triangulatedMesh>& simplifiedRobotLinkMeshes = std::get<0>(result);
	std::vector<uint32_t>& meshLinkIndices = std::get<1>(result);

	const OpenRAVE::EnvironmentBasePtr env = robot->GetEnv();
	OpenRAVE::EnvironmentMutex::scoped_lock lock(env->GetMutex()); // lock environment
	const std::vector<OpenRAVE::KinBody::LinkPtr> links = robot->GetLinks();

#if 1 //use my own code (can't get other packages to work on these meshes -- is it maybe because triangle vertex orderings are inconsistent?)
	/*
	 * remove triangles facing centroid of each mesh and remove small triangles (these criteria are specific to openrave robot models)
	 *
	 * 20131203: the triangles in the openrave robot models aren't consistent--neighboring triangles sometimes have opposite vertex orderings
	 */
{
	for(size_t j = 0; j < links.size(); j++)
	{
		const OpenRAVE::TriMesh& linkMesh = links[j]->GetCollisionData();
		cout << "initial mesh: " << linkMesh.vertices.size() << ", " << (linkMesh.indices.size() / 3) << endl;

		const OpenRAVE::AABB bbox = links[j]->ComputeAABB();
		const rgbd::eigen::Vector3f linkCentroid(bbox.pos.x, bbox.pos.y, bbox.pos.z);

		std::vector<size_t> trianglesToInclude;
		for(size_t i = 0; i < linkMesh.indices.size() / 3; i++)
		{
			const std::array<rgbd::eigen::Vector3f, 3> xs =
			{
				rgbd::eigen::Vector3f(linkMesh.vertices[linkMesh.indices[i * 3 + 0]].x, linkMesh.vertices[linkMesh.indices[i * 3 + 0]].y, linkMesh.vertices[linkMesh.indices[i * 3 + 0]].z),
				rgbd::eigen::Vector3f(linkMesh.vertices[linkMesh.indices[i * 3 + 1]].x, linkMesh.vertices[linkMesh.indices[i * 3 + 1]].y, linkMesh.vertices[linkMesh.indices[i * 3 + 1]].z),
				rgbd::eigen::Vector3f(linkMesh.vertices[linkMesh.indices[i * 3 + 2]].x, linkMesh.vertices[linkMesh.indices[i * 3 + 2]].y, linkMesh.vertices[linkMesh.indices[i * 3 + 2]].z)
			};
			float maxDim = -1;
			for(size_t k = 0; k < 3; k++)
				for(size_t l = 0; l < 3; l++)
				{
					const float dim = fabs(xs[k][l] - xs[(k + 1) % 3][l]);
					if(dim > maxDim) maxDim = dim;
				}
			if(maxDim > params.triDimThreshold) //check for reasonable size; don't use area because many of the triangles in these models are almost degenerate
			{
				const rgbd::eigen::Vector3f normal = (xs[1] - xs[0]).cross(xs[2] - xs[0]).normalized();
			//	if(normal.dot(linkCentroid - xs[0]) <= 0) //if normal points away from centroid
					trianglesToInclude.push_back(i);
			}
		}

		simplifiedRobotLinkMeshes.resize(simplifiedRobotLinkMeshes.size() + 1);
		triangulatedMesh& simplifiedMesh = simplifiedRobotLinkMeshes.back();
		meshLinkIndices.push_back(j);

		simplifiedMesh.allocateVertices(trianglesToInclude.size() * 3);
		simplifiedMesh.allocateTriangles(trianglesToInclude.size());
		cout << "final: " << simplifiedMesh.numVertices() << ", " << simplifiedMesh.numTriangles() << endl;
		for(size_t i = 0; i < trianglesToInclude.size(); i++)
		{
			for(size_t k = 0; k < 3; k++)
			{
				rgbd::pt pt;
				pt.x = linkMesh.vertices[linkMesh.indices[trianglesToInclude[i] * 3 + k]].x;
				pt.y = linkMesh.vertices[linkMesh.indices[trianglesToInclude[i] * 3 + k]].y;
				pt.z = linkMesh.vertices[linkMesh.indices[trianglesToInclude[i] * 3 + k]].z;
				simplifiedMesh.v(i * 3 + k) = pt;
			}

			triangulatedMesh::triangle tri;
			for(size_t k = 0; k < 3; k++) tri.v[k] = 3 * i + k;
			simplifiedMesh.setTriangle(i, tri);
		}
	}

#if 0 //visualize simplified mesh
	triangulatedMesh visMesh;
	for(size_t j = 0; j < links.size(); j++) visMesh.append(simplifiedRobotLinkMeshes[j]);
	for(size_t j = 0; j < visMesh.numVertices(); j++) visMesh.v(j).rgb = rgbd::packRGB(200, 220, 255);
	visMesh.writePLY("robotLinksSimplified.ply");
	exit(0);
#endif
}
#else
	/*
	 * use generic mesh simplification software (20131126 I haven't been able to get either cgal or progressiveMeshes to work, basically, at all)
	 */
{
	for(size_t j = 0; j < links.size(); j++)
	{
		const OpenRAVE::TriMesh& linkMesh = links[j]->GetCollisionData();
		cout << "initial mesh: " << linkMesh.vertices.size() << ", " << (linkMesh.indices.size() / 3) << endl;

#ifdef USE_PROGMESH
		Quadrics meshSimplifier;
		meshSimplifier.createFromRAVEMesh(linkMesh);
		meshSimplifier.initial_quadrics();
		const size_t targetNumFaces = .9 * (linkMesh.indices.size() / 3); //TODO ?
		meshSimplifier.construct_n_contract(targetNumFaces);

		simplifiedRobotLinkMeshes[j].allocateVertices(meshSimplifier.vertices.size());
		simplifiedRobotLinkMeshes[j].allocateTriangles(meshSimplifier.faces.size());
		cout << "final: " << simplifiedRobotLinkMeshes[j].numVertices() << ", " << simplifiedRobotLinkMeshes[j].numTriangles() << endl;
		size_t i = 0;
		std::unordered_map<int, int> imap(meshSimplifier.vertices.size());
		for(auto k : meshSimplifier.vertices) //TODO don't copy vertices that are no longer used in any triangle
		{
			rgbd::pt pt;
			pt.x = k.second.x;
			pt.y = k.second.y;
			pt.z = k.second.z;
			simplifiedRobotLinkMeshes[j].v(i) = pt;
			imap[k.first] = i;
			i++;
		}
		for(size_t i = 0; i < meshSimplifier.faces.size(); i++)
		{
			triangulatedMesh::triangle tri;
			for(size_t k = 0; k < 3; k++)
			{
				tri.v[k] = imap[meshSimplifier.faces[i].id_vertex[k]];
				ASSERT_ALWAYS(tri.v[k] < simplifiedRobotLinkMeshes[j].numVertices());
			}
			simplifiedRobotLinkMeshes[j].setTriangle(i, tri);
		}
#else //use cgal
		/*
		 * build cgal polyhedron from rave mesh, then do mesh simplification in cgal
		 */

		the function won't collapse any edge whose triangles form more than a 1-degree angle! this is useless!

		typedef CGAL::Polyhedron_3<Kernel, CGAL::Polyhedron_items_with_id_3> SurfaceT;

		struct polyhedronBuilderFromTriangles : public CGAL::Modifier_base<SurfaceT::HalfedgeDS>
		{
			polyhedronBuilderFromTriangles(const OpenRAVE::TriMesh& m) : raveMesh(m) {}

			// Postcondition: hds is a valid polyhedral surface
			void operator () (SurfaceT::HalfedgeDS& hds)
			{
				CGAL::Polyhedron_incremental_builder_3<SurfaceT::HalfedgeDS> B(hds, true);
				B.begin_surface(raveMesh.vertices.size(), raveMesh.indices.size() / 3);
				typedef typename SurfaceT::HalfedgeDS::Vertex Vertex;
				typedef typename Vertex::Point Point;
				for(size_t i = 0; i < raveMesh.vertices.size(); i++) B.add_vertex(Point(raveMesh.vertices[i].x, raveMesh.vertices[i].y, raveMesh.vertices[i].z));
				for(size_t i = 0; i < raveMesh.indices.size() / 3; i++)
				{
					B.begin_facet();
					for(size_t k = 0; k < 3; k++) B.add_vertex_to_facet(i * 3 + k);
					B.end_facet();
				}
				B.end_surface();
			}

			const OpenRAVE::TriMesh& raveMesh;
		};

		namespace SMS = CGAL::Surface_mesh_simplification;

		struct My_visitor : public SMS::Edge_collapse_visitor_base<SurfaceT>
		{
		// Called during the collecting phase for each edge collected.
		void OnCollected( Profile const&, boost::optional<double> const& )
		{
		cout << "collected" << endl;
		}
		// Called during the processing phase for each edge selected.
		// If cost is absent the edge won't be collapsed.
		void OnSelected(Profile const&
		,boost::optional<double> cost
		,std::size_t initial
		,std::size_t current
		)
		{
		cout << "selected" << endl;
		}
		// Called during the processing phase for each edge being collapsed.
		// If placement is absent the edge is left uncollapsed.
		void OnCollapsing(Profile const&
		,boost::optional<Point> placement
		)
		{
			cout << "collapsing" << endl;
		}
		// Called for each edge which failed the so called link-condition,
		// that is, which cannot be collapsed because doing so would
		// turn the surface into a non-manifold.
		void OnNonCollapsable( Profile const& )
		{
			cout << "noncollapsible" << endl;
		}
		// Called AFTER each edge has been collapsed
		void OnCollapsed( Profile const&, SurfaceT::HalfedgeDS::Vertex_handle )
		{
			cout << "collapsed" << endl;
		}
		};

		My_visitor vis;

		SurfaceT surface;
		polyhedronBuilderFromTriangles builder(linkMesh);
		surface.delegate(builder); //build polyhedron

#if 0
		 // Write polyhedron in Object File Format (OFF).
		typedef SurfaceT::Facet_iterator Facet_iterator;
		typedef SurfaceT::Halfedge_around_facet_circulator Halfedge_facet_circulator;
		std::ofstream outfile("poly.off");
		CGAL::set_ascii_mode( outfile);
		outfile << "OFF" << std::endl << surface.size_of_vertices() << ' '
		<< surface.size_of_facets() << " 0" << std::endl;
		std::copy( surface.points_begin(), surface.points_end(),
		std::ostream_iterator<Point>( outfile, "\n"));
		for ( Facet_iterator i = surface.facets_begin(); i != surface.facets_end(); ++i) {
		Halfedge_facet_circulator j = i->facet_begin();
		ASSERT_ALWAYS( CGAL::circulator_size(j) == 3);
		outfile << CGAL::circulator_size(j) << ' ';
		do {
			outfile << ' ' << std::distance(surface.vertices_begin(), j->vertex());
		} while ( ++j != i->facet_begin());
		outfile << std::endl;
		}
		outfile.close();
		exit(0);
#endif

		 // The items in this polyhedron have an "id()" field which the default index maps used in the algorithm need to get the index of a vertex/edge.
		// However, the Polyhedron_3 class doesn't assign any value to this id(), so we must do it here.
		int index = 0 ;
		for( SurfaceT::Halfedge_iterator eb = surface.halfedges_begin(), ee = surface.halfedges_end() ; eb != ee; ++ eb)
			eb->id() = index++;
		index = 0 ;
		for( SurfaceT::Vertex_iterator vb = surface.vertices_begin(), ve = surface.vertices_end(); vb != ve; ++ vb)
			vb->id() = index++;
		const SMS::Count_ratio_stop_predicate<SurfaceT> stop(0.1); // simplification stops when the number of undirected edges drops below N% of the initial count
		SMS::edge_collapse(surface, stop, /*CGAL::visitor(vis)*//*, CGAL::get_cost(SMS::Edge_length_cost<SurfaceT>()).get_placement(SMS::Midpoint_placement<SurfaceT>())*/);

		simplifiedRobotLinkMeshes[j].allocateVertices(surface.size_of_vertices());
		simplifiedRobotLinkMeshes[j].allocateTriangles(surface.size_of_facets());
		cout << "final: " << simplifiedRobotLinkMeshes[j].numVertices() << ", " << simplifiedRobotLinkMeshes[j].numTriangles() << endl;
		size_t i = 0;
		for(auto v = surface.vertices_begin(); v != surface.vertices_end(); v++, i++)
		{
			const auto cgalpt = v->point();
			rgbd::pt pt;
			pt.x = cgalpt.x();
			pt.y = cgalpt.y();
			pt.z = cgalpt.z();
			simplifiedRobotLinkMeshes[j].v(i) = pt;
		}
		i = 0;
		for(auto t = surface.facets_begin(); t != surface.facets_end(); t++, i++)
		{
			ASSERT_ALWAYS(t->is_triangle());
			SurfaceT::Halfedge_around_facet_circulator c = t->facet_begin();
			triangulatedMesh::triangle tri;
			for(size_t k = 0; k < 3; k++, c++) {tri.v[k] = std::distance(surface.vertices_begin(), c->vertex());//c->vertex()->id();
			cout << std::distance(surface.vertices_begin(), c->vertex()) << ' ' << c->vertex()->id() << "   ";
			}
			cout << endl;
			simplifiedRobotLinkMeshes[j].setTriangle(i, tri);
		}
#endif
	}
#if 1 //visualize simplified mesh
	triangulatedMesh visMesh;
	for(size_t j = 0; j < links.size(); j++) visMesh.append(simplifiedRobotLinkMeshes[j]);
	for(size_t j = 0; j < visMesh.numVertices(); j++) visMesh.v(j).rgb = rgbd::packRGB(200, 220, 255);
	visMesh.writePLY("robotLinksSimplified.ply");
#endif
	exit(0);
}
#endif

	return result;
}
