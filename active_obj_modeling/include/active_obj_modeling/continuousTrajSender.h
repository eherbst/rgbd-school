/*
 * continuousTrajSender: wrapper for trajectory executor to allow continual replanning
 *
 * Evan Herbst
 * 12 / 10 / 13
 */

#ifndef EX_CONTINUOUS_TRAJECTORY_SENDER_H
#define EX_CONTINUOUS_TRAJECTORY_SENDER_H

#include <vector>
#include <deque>
#include <mutex>
#include <boost/signals2/signal.hpp>
#include <openrave-core.h>
#include "active_vision_common/cameraOnRobotHandler.h"

/*
 * to be subclassed by robot-specific classes
 */
class continuousTrajSender
{
	public:

		typedef std::vector<OpenRAVE::dReal> WaypointT; //joint values
		struct trajspec
		{
			int64_t id;
			std::vector<WaypointT> waypts; //dof values at each timestep
			float deltaT; //between waypoints
		};

		/*
		 * if simulationEnv, we'll run everything in simulation
		 */
		continuousTrajSender(const std::shared_ptr<cameraOnRobotHandler>& camHandler, const OpenRAVE::EnvironmentBasePtr& simulationEnv);
		virtual ~continuousTrajSender() {}

		/*
		 * to be run in its own thread; will run forever
		 *
		 * poll for new trajectories and send parts of them to a robot as appropriate
		 */
		virtual void run() = 0;

		/*
		 * trajID: client-defined
		 */
		void addTrajectory(const int64_t trajID, const float /*unused*/, const std::vector<WaypointT>& traj, const float deltaT);

		void addTrajCompletionCallback(const std::function<void (const int64_t trajID, const bool success)>& f);

		/*
		 * the # of trajectories in the queue after the one currently executing
		 */
		size_t numTrajsWaiting() const;
		/*
		 * in seconds
		 */
		std::vector<float> waitingTrajDurations() const;

		/*
		 * return the end of the current trajectory, or empty if no trajectory has been requested
		 */
		WaypointT getCurGoalConfiguration() const;

	protected:

		std::shared_ptr<cameraOnRobotHandler> robotCamHandler;
		bool runInSimulation; //if true, simulate everything in openrave
		OpenRAVE::EnvironmentBasePtr simulationEnv; //valid iff runInSimulation

		std::deque<trajspec> trajs; //list of (start time, traj)
		mutable std::mutex trajsMux;

		WaypointT curTrajEndConfiguration;
		mutable std::mutex curGoalMux;

		boost::signals2::signal<void (const int64_t trajID, const bool success)> trajCompletionSignal; //called after attempting to run a trajectory and let it finish
};

#endif //header
