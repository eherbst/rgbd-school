/*
 * see whether pcl's integral-image normals will give us any speedup over our cpu-based standard nearest-nbr pca normal estimation
 * (see http://pointclouds.org/documentation/tutorials/normal_estimation_using_integral_images.php)
 *
 * 20131104 verdict: it doesn't
 *
 * Evan Herbst
 * 11 / 4 / 13
 */

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/impl/pcl_base.hpp>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/impl/integral_image_normal.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include "rgbd_util/timer.h"
#include "pcl_rgbd/pointTypes.h"
#include "pcl_rgbd/cloudNormals.h"
#include "rgbd_bag_utils/readOneCloud.h"

int
main (int argc, char* argv[])
{
	 // load point cloud
	pcl::PointCloud<rgbd::pt>::Ptr cloud(new pcl::PointCloud<rgbd::pt>);
	readCloud(*cloud, argv[1]);
	pcl::PointCloud<rgbd::pt> orgcloud;
	orgcloud.height = 480;
	orgcloud.width = 640;
	boost::multi_array<bool, 2> validity;
	rgbd::computeOrganizedPointCloud(*cloud, orgcloud, validity);
	*cloud = orgcloud;

	 // estimate normals
	 pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

	 rgbd::timer t;
	 pcl::IntegralImageNormalEstimation<rgbd::pt, pcl::Normal> ne;
	 ne.setNormalEstimationMethod (ne.AVERAGE_DEPTH_CHANGE);//AVERAGE_3D_GRADIENT);
	 ne.setMaxDepthChangeFactor(0.02f);
	 ne.setNormalSmoothingSize(5/*10*/);
	 ne.setInputCloud(cloud);
	 ne.compute(*normals);
	 t.stop("compute");

	 // visualize normals
	 pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	 viewer.setBackgroundColor (0.0, 0.0, 0.5);
	 viewer.addPointCloudNormals<rgbd::pt,pcl::Normal>(cloud, normals);

	 while (!viewer.wasStopped ())
	 {
		viewer.spinOnce ();
	 }
	 return 0;
}
