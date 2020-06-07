/*
 * continuousTrajSender: wrapper for trajectory executor to allow continual replanning
 *
 * Evan Herbst
 * 12 / 10 / 13
 */

#include <iostream>
#include "rgbd_util/assert.h"
#include "active_obj_modeling/continuousTrajSender.h"
using std::cout;
using std::endl;

/*
 * if simulationEnv, we'll run everything in simulation
 */
continuousTrajSender::continuousTrajSender(const std::shared_ptr<cameraOnRobotHandler>& camHandler, const OpenRAVE::EnvironmentBasePtr& simulationEnv)
: robotCamHandler(camHandler), runInSimulation(simulationEnv), simulationEnv(simulationEnv)
{
}

void continuousTrajSender::addTrajCompletionCallback(const std::function<void (const int64_t trajID, const bool success)>& f)
{
	trajCompletionSignal.connect(f);
}

/*
 * trajID: client-defined
 */
void continuousTrajSender::addTrajectory(const int64_t trajID, const float /* unused */, const std::vector<WaypointT>& traj, const float deltaT)
{
	ASSERT_ALWAYS(traj.size() > 1); //avoid nonsensical trajectories

	std::lock_guard<std::mutex> lock(trajsMux);
	trajs.resize(trajs.size() + 1);
	trajspec& t = trajs.back();
	t.id = trajID;
	t.waypts = traj;
	t.deltaT = deltaT;
//	//sort by start time increasing
//	std::inplace_merge(trajs.begin(), trajs.end() - 1, trajs.end(), [](const std::pair<float, std::vector<WaypointT>>& t1, const std::pair<float, std::vector<WaypointT>>& t2){return t1.first < t2.first;});
}

/*
 * the # of trajectories in the queue after the one currently executing
 */
size_t continuousTrajSender::numTrajsWaiting() const
{
	std::lock_guard<std::mutex> lock(trajsMux);
	return trajs.size(); //assuming each one gets popped when it starts running
}
/*
 * in seconds
 */
std::vector<float> continuousTrajSender::waitingTrajDurations() const
{
	std::lock_guard<std::mutex> lock(trajsMux);
	std::vector<float> durations(trajs.size());
	for(size_t i = 0; i < trajs.size(); i++) durations[i] = trajs[i].deltaT * (trajs[i].waypts.size() - 1);
	return durations;
}

/*
 * return the end of the current trajectory, or empty if no trajectory has been requested
 */
continuousTrajSender::WaypointT continuousTrajSender::getCurGoalConfiguration() const
{
	std::lock_guard<std::mutex> lock(curGoalMux);
	return curTrajEndConfiguration;
}
