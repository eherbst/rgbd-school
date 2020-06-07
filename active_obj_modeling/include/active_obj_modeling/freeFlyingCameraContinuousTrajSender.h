/*
 * freeFlyingCameraTrajSender: wrapper for trajectory executor to allow continual replanning
 *
 * Evan Herbst
 * 1 / 21 / 14
 */

#ifndef EX_FREE_FLYING_CONTINUOUS_TRAJ_SENDER_H
#define EX_FREE_FLYING_CONTINUOUS_TRAJ_SENDER_H

#include "active_vision_common/robotSpec.h"
#include "active_obj_modeling/continuousTrajSender.h"

class freeFlyingCameraContinuousTrajSender: public continuousTrajSender
{
	public:

		/*
		 * if simulationEnv, we'll run everything in simulation
		 */
		freeFlyingCameraContinuousTrajSender(const std::shared_ptr<cameraOnRobotHandler>& camHandler, const std::shared_ptr<robotSpec>& interface, const OpenRAVE::EnvironmentBasePtr& simulationEnv);
		virtual ~freeFlyingCameraContinuousTrajSender() {}

		/*
		 * to be run in its own thread; will run forever
		 *
		 * poll for new trajectories and send parts of them to a robot as appropriate
		 */
		virtual void run();

	private:

		std::shared_ptr<robotSpec> robotInterface;
};

#endif //header
