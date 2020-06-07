/*
 * collisionChecking: we seem to be able to do it much more quickly ourselves than openrave can
 *
 * Evan Herbst
 * 12 / 20 / 13
 */

#include <iostream>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/internal/Intersections_3/Bbox_3_Triangle_3_do_intersect.h>
#include <CGAL/Bbox_3.h>
#include "rgbd_util/timer.h"
#include "rgbd_util/parallelism.h"
#include "rgbd_util/threadPool.h"
#include "openrave_utils/openraveUtils.h"
#include "active_obj_modeling/collisionChecking.h"
using std::cout;
using std::endl;

//copied from AABB_triangle_primitive.h and edited by EVH
namespace CGAL {

/*
 * add stuff AABB_tree needs in an element type to Bbox_3
 */
class evhBbox_3 : public Bbox_3
{
	public:

		evhBbox_3() {}
		evhBbox_3(const Bbox_3& b) : Bbox_3(b) {}

		Bbox_3 bbox() const {return *this;}
};

#ifdef UNUSED
template <class GeomTraits, class Iterator>
class AABB_box_primitive
{
public:
  // types
  typedef Iterator Id; // Id type
  typedef typename GeomTraits::Point_3 Point; // point type
  typedef typename CGAL::evhBbox_3 Datum; // datum type

private:
  // member data
  Id m_it; // iterator
  Datum m_datum; //primitive

  // constructor
public:
  AABB_box_primitive() {}
  AABB_box_primitive(const Id& id)
  {
	  m_datum = *id;
	  m_it = id;
  }
  AABB_box_primitive(const AABB_box_primitive& primitive)
  {
		m_datum = primitive.datum();
		m_it = primitive.id();
  }
public:
  Id& id() { return m_it; }
  const Id& id() const { return m_it; }
  Datum& datum() { return m_datum; }
  const Datum& datum() const { return m_datum; }

  /// Returns a point on the primitive
  Point reference_point() const { return Point(m_datum.xmin(), m_datum.ymin(), m_datum.zmin()); }
};
#endif

}  // end namespace CGAL

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point;
typedef Kernel::Triangle_3 Triangle;
typedef CGAL::AABB_triangle_primitive<Kernel, std::vector<Triangle>::iterator, CGAL::Tag_true/* whether to cache vertices in the aabbtree */> TriPrimitive;
typedef CGAL::AABB_traits<Kernel, TriPrimitive> AABB_triangle_traits;
#ifdef UNUSED
typedef CGAL::AABB_box_primitive<Kernel, std::vector<CGAL::Bbox_3>::iterator> BoxPrimitive;
typedef CGAL::AABB_traits<Kernel, BoxPrimitive> AABB_box_traits;
#endif

struct precomputedEnvCollisionData
{
	std::shared_ptr<CGAL::AABB_tree<AABB_triangle_traits>> aabbTree;
};

std::shared_ptr<precomputedEnvCollisionData> precomputeForCollisionChecking(const std::vector<std::shared_ptr<triangulatedMesh>>& envMeshes,
	const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& envMeshPosesWrtRaveWorld)
{
	std::shared_ptr<precomputedEnvCollisionData> envData(new precomputedEnvCollisionData);

	rgbd::timer u;
	/*
	 * create a mesh of all obstacles
	 */
	std::vector<Triangle> trianglesV;
	for(size_t j = 0; j < envMeshes.size(); j++)
	{
		const size_t nT = trianglesV.size();
		trianglesV.resize(trianglesV.size() + envMeshes[j]->numTriangles());
		const std::vector<triangulatedMesh::triangle>& tris = envMeshes[j]->getTriangles();
		for(size_t i = 0; i < envMeshes[j]->numTriangles(); i++)
		{
			//vertices in rave world coords
			const rgbd::eigen::Vector4f p0 = envMeshPosesWrtRaveWorld[j] * rgbd::ptX2eigen<rgbd::eigen::Vector4f>(envMeshes[j]->v(tris[i].v[0])),
												p1 = envMeshPosesWrtRaveWorld[j] * rgbd::ptX2eigen<rgbd::eigen::Vector4f>(envMeshes[j]->v(tris[i].v[1])),
												p2 = envMeshPosesWrtRaveWorld[j] * rgbd::ptX2eigen<rgbd::eigen::Vector4f>(envMeshes[j]->v(tris[i].v[2]));
			trianglesV[nT + i] = Triangle(Point(p0.x(), p0.y(), p0.z()),
													Point(p1.x(), p1.y(), p1.z()),
													Point(p2.x(), p2.y(), p2.z()));
		}
	}
	u.stop("list tris for aabbtree");
	u.restart();
	envData->aabbTree.reset(new CGAL::AABB_tree<AABB_triangle_traits>(trianglesV.begin(), trianglesV.end()));
	envData->aabbTree->build(); //if we don't call build() explicitly, the tree will actually be built upon the first search query; this is here so timing numbers will make sense
	u.stop("build aabbtree");

	return envData;
}

/*
 * return whether each robot configuration puts the robot in collision with the environment
 *
 * output is undefined if a configuration is the empty vector
 *
 * use raveEnvs for multithreading
 *
 * robotLinkMeshLinkIndices: which link each mesh comes from
 */

std::vector<uint8_t> checkRAVERobotCollisions(const std::vector<OpenRAVE::EnvironmentBasePtr>& raveEnvs, const std::string& robotName, robotSpec& robotInterface, const std::vector<std::shared_ptr<triangulatedMesh>>& envMeshes,
	const std::vector<rgbd::eigen::Affine3f, rgbd::eigen::aligned_allocator<rgbd::eigen::Affine3f>>& envMeshPosesWrtRaveWorld, const std::vector<triangulatedMesh>& robotLinkMeshes, const std::vector<uint32_t>& robotLinkMeshLinkIndices,
	const std::vector<std::vector<OpenRAVE::dReal>>& configurations)
{
	std::shared_ptr<precomputedEnvCollisionData> envData = precomputeForCollisionChecking(envMeshes, envMeshPosesWrtRaveWorld);
	return checkRAVERobotCollisions(raveEnvs, robotName, robotInterface, envData, robotLinkMeshes, robotLinkMeshLinkIndices, configurations);
}

std::vector<uint8_t> checkRAVERobotCollisions(const std::vector<OpenRAVE::EnvironmentBasePtr>& raveEnvs, const std::string& robotName, robotSpec& robotInterface, const std::shared_ptr<precomputedEnvCollisionData>& envData,
	const std::vector<triangulatedMesh>& robotLinkMeshes, const std::vector<uint32_t>& robotLinkMeshLinkIndices, const std::vector<std::vector<OpenRAVE::dReal>>& configurations)
{
	std::vector<uint8_t> collisionFlags(configurations.size(), true);
	const auto& envAABBTree = *envData->aabbTree;
	rgbd::timer u;

	std::vector<std::vector<OpenRAVE::Transform>> raveLinkPosesWrtRaveWorld(configurations.size());
	std::vector<std::vector<OpenRAVE::AABB>> raveLinkBBoxes(configurations.size()); //each is empty if we find a robot-robot collision; if not empty we can later check robot-env collision
{
	const size_t numThreads = 1;//TODO try raveEnvs.size();
	rgbd::threadGroup tg(numThreads);
	for(size_t m = 0; m < numThreads; m++)
		tg.addTask([&,m]()
			{
				OpenRAVE::RobotBasePtr robot = raveEnvs[m]->GetRobot(robotName);
				const std::vector<OpenRAVE::KinBody::LinkPtr>& links = robot->GetLinks();
				for(size_t i = m; i < configurations.size(); i += numThreads)
					if(!configurations[i].empty())
					{
						robotInterface.setRAVERobotConfiguration(robot, configurations[i]);
						if(!robot->CheckSelfCollision()) //do robot-robot collision checking here and only do robot-env checking below; TODO could we speed up the robot-robot part too?
						{
							robot->GetLinkTransformations(raveLinkPosesWrtRaveWorld[i]);
							raveLinkBBoxes[i].resize(links.size());
							for(size_t j = 0; j < links.size(); j++)
							{
								raveLinkBBoxes[i][j] = links[j]->ComputeAABB();
							}
						}
					}
			});
	tg.wait();
}
	u.stop("get boxes & xforms");

	u.restart();
{
	const size_t numThreads = getSuggestedThreadCount(2, 2);
	rgbd::threadGroup tg(numThreads);
	for(size_t m = 0; m < numThreads; m++)
		tg.addTask([&,m]()
			{
				for(size_t i = m; i < configurations.size(); i += numThreads) //non-colliding poses will probably occur in bunches; try to spread them over threads
				{
					if(!configurations[i].empty() && !raveLinkPosesWrtRaveWorld[i].empty())
					{
						std::vector<rgbd::eigen::Affine3f> linkPosesWrtRaveWorld(raveLinkPosesWrtRaveWorld[i].size());
						for(size_t j = 0; j < raveLinkPosesWrtRaveWorld[i].size(); j++) linkPosesWrtRaveWorld[j] = raveXform2eigenXform(raveLinkPosesWrtRaveWorld[i][j]);

						bool collision = false;
						for(int32_t j = robotLinkMeshes.size() - 1; j >= 0 && !collision; j--) //the outermost links are the most likely to collide, so try them first -- gets us a very small speedup
						{
							/*
							 * check mesh bbox before checking thousands of triangles (gets us about a 2.5x speedup)
							 */
							const OpenRAVE::RaveVector<OpenRAVE::dReal> mins = raveLinkBBoxes[i][j].pos - raveLinkBBoxes[i][j].extents, maxes = raveLinkBBoxes[i][j].pos + raveLinkBBoxes[i][j].extents;
							//const CGAL::evhBbox_3 bbox(CGAL::Bbox_3(mins.x, mins.y, mins.z, maxes.x, maxes.y, maxes.z));
							const CGAL::Bbox_3 bbox(mins.x, mins.y, mins.z, maxes.x, maxes.y, maxes.z);
							if(envAABBTree.do_intersect(bbox))
							{
								const triangulatedMesh& linkMesh = robotLinkMeshes[j];
								const uint32_t meshLinkIndex = robotLinkMeshLinkIndices[j];
								const auto& tris = linkMesh.getTriangles();
								for(size_t m = 0; m < linkMesh.numTriangles(); m++) //for each triangle
								{
									const rgbd::eigen::Vector3f xformedPt0 = linkPosesWrtRaveWorld[meshLinkIndex] * rgbd::ptX2eigen<rgbd::eigen::Vector3f>(linkMesh.v(tris[m].v[0])),
										xformedPt1 = linkPosesWrtRaveWorld[meshLinkIndex] * rgbd::ptX2eigen<rgbd::eigen::Vector3f>(linkMesh.v(tris[m].v[1])),
										xformedPt2 = linkPosesWrtRaveWorld[meshLinkIndex] * rgbd::ptX2eigen<rgbd::eigen::Vector3f>(linkMesh.v(tris[m].v[2]));
									const Triangle tri(Point(xformedPt0.x(), xformedPt0.y(), xformedPt0.z()), Point(xformedPt1.x(), xformedPt1.y(), xformedPt1.z()), Point(xformedPt2.x(), xformedPt2.y(), xformedPt2.z()));
									if(envAABBTree.do_intersect(tri))
									{
										collision = true;
										break;
									}
								}
							}
						}
						if(!collision) collisionFlags[i] = false;
					}
				}
			});
	tg.wait();
}
	u.stop("do collision-checking main loop");
	return collisionFlags;
}
