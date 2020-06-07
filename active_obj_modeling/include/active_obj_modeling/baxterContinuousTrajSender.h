/*
 * baxterContinuousTrajSender: wrapper for trajectory executor to allow continual replanning
 *
 * Evan Herbst
 * 1 / 1 / 14
 */

#ifndef EX_BAXTER_CONTINUOUS_TRAJ_SENDER_H
#define EX_BAXTER_CONTINUOUS_TRAJ_SENDER_H

#include <memory>
#include <ros/node_handle.h>
#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include "active_obj_modeling/continuousTrajSender.h"

class baxterContinuousTrajSender: public continuousTrajSender
{
	public:

		/*
		 * if simulationEnv, we'll run everything in simulation
		 */
		baxterContinuousTrajSender(const std::shared_ptr<cameraOnRobotHandler>& camHandler, const OpenRAVE::EnvironmentBasePtr& simulationEnv);
		virtual ~baxterContinuousTrajSender() {}

		/*
		 * to be run in its own thread; will run forever
		 *
		 * poll for new trajectories and send parts of them to a robot as appropriate
		 */
		virtual void run();

	private:

		ros::NodeHandle nh_global;
		std::shared_ptr<actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>> runTrajClient;
};

#endif //header
