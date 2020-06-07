/*
 * marvinContinuousTrajSender: wrapper for trajectory executor to allow continual replanning
 *
 * Evan Herbst
 * 1 / 1 / 14
 */

#ifndef EX_MARVIN_CONTINUOUS_TRAJ_SENDER_H
#define EX_MARVIN_CONTINUOUS_TRAJ_SENDER_H

#include <memory>
#include <ros/node_handle.h>
#include <ros/service_client.h>
#include "active_obj_modeling/continuousTrajSender.h"

class marvinContinuousTrajSender: public continuousTrajSender
{
	public:

		/*
		 * if simulationEnv, we'll run everything in simulation
		 */
		marvinContinuousTrajSender(const std::shared_ptr<cameraOnRobotHandler>& camHandler, const OpenRAVE::EnvironmentBasePtr& simulationEnv);
		virtual ~marvinContinuousTrajSender() {}

		/*
		 * to be run in its own thread; will run forever
		 *
		 * poll for new trajectories and send parts of them to a robot as appropriate
		 */
		virtual void run();

	private:

		std::shared_ptr<ros::NodeHandle> nh_global;
		ros::ServiceClient runTrajectoryServiceClient;
};

#endif //header
