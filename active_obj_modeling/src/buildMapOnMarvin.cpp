/*
 * buildMapOnMarvin: have the user move the arm with attached camera around so marvin will know where around it it's safe to move the arm for active vision later
 *
 * Evan Herbst
 * 10 / 9 / 13
 */

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include "rgbd_util/assert.h"
#include "rgbd_util/config.h"
#include "openrave_utils/openraveUtils.h"
#include "active_vision_common/cameraOnRobotROSHandler.h"
using std::cout;
using std::endl;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

/*
 * README how to use:
 * - put arm into zero-gravity mode
 * - move arm to good starting pose for mapping (where the camera can see something it can align to)
 * - run this driver
 * - kill this driver when done recording (it should finalize the bag then)
 */
void recordWithMarvin(int argc, char* argv[], const fs::path& outdir)
{
	ros::init(argc, argv, "buildMapOnMarvin", ros::init_options::AnonymousName);

	configOptions cfg;

	//for cameraOverROSHandler
	//TODO figure out how to get compressed imgs with timestamps that match (20131009 if you just use the /compressed topic here, they don't) so we can use a time synchronizer
	cfg.set("imgTopic", "/camera/rgb/image_color_sync");
	cfg.set("depthTopic", "/camera/depth/image_sync");
	cfg.set("spinInSeparateThread", false);

	//for cameraOnRobotHandler
	cfg.set("dumpRGBDJOutpath", outdir / "rgbdj.bag"); //tell it to dump to disk

	//for openraveUtils
	cfg.set("raveBasePath", "/home/eherbst/proj/ros-extras/openrave_planning/openrave/share/openrave-latest");

	cfg.set("initRave", true);
	cfg.set("initEnv", true);
	OpenRAVE::EnvironmentBasePtr raveEnv;
	loadRaveRobot(raveEnv, cfg);

	cameraOnRobotROSHandler recorder(cfg, raveEnv); //register some listeners
	recorder.init();
	ros::spin();
}

int main(int argc, char* argv[])
{
	po::options_description desc("Options");
	desc.add_options()
		("mode,m", po::value<std::string>()->default_value("a"), "r = recording | a = aggregating")
		("outdir,o", po::value<fs::path>(), "for recording; will be created if nec")
		("inbag,i", po::value<fs::path>(), "previously recorded, for aggregating; will be created if nec")
		;
	po::variables_map vars;
	po::store(po::command_line_parser(argc, argv).options(desc).run(), vars);
	po::notify(vars);

	const std::string mode = vars["mode"].as<std::string>();
	if(mode == "r") //record and dump a bag with rgbd frames and camera xforms wrt robot base
	{
		const fs::path outdir = vars["outdir"].as<fs::path>();
		fs::create_directories(outdir);

		recordWithMarvin(argc, argv, outdir);
	}
	else if(mode == "a") //build a map using a recorded file
	{
		const fs::path inbagpath = vars["inbag"].as<fs::path>();
		rosbag::Bag inbag(inbagpath.string());
		//get first joint angles and dump to disk
		rosbag::View view(inbag, rosbag::TypeQuery("pr_msgs/WAMJointState"));
		for(const rosbag::MessageInstance& m : view)
		{
			const pr_msgs::WAMJointState::ConstPtr joints = m.instantiate<pr_msgs::WAMJointState>();
			if(joints)
			{
				cout << "initial joints: "; std::copy(joints->positions.begin(), joints->positions.end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
				break;
			}
		}
		inbag.close();

		//TODO run peter mapping on the rest of the bag
	}
	else throw std::invalid_argument("invalid mode string");

	return 0;
}
