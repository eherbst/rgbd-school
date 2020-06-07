/*
 * freeFlyingCameraTrajSender: wrapper for trajectory executor to allow continual replanning
 *
 * Evan Herbst
 * 1 / 21 / 14
 */

#include <iostream>
#include <thread>
#include "grasp_utils/ExecuteGraspTest.h" //the service to run a trajectory
#include "active_vision_common/openraveUtils.h"
#include "active_obj_modeling/freeFlyingCameraContinuousTrajSender.h"
using std::cout;
using std::endl;

/*
 * if simulationEnv, we'll run everything in simulation
 */
freeFlyingCameraContinuousTrajSender::freeFlyingCameraContinuousTrajSender(const std::shared_ptr<cameraOnRobotHandler>& camHandler, const std::shared_ptr<robotSpec>& interface, const OpenRAVE::EnvironmentBasePtr& simulationEnv)
: continuousTrajSender(camHandler, simulationEnv), robotInterface(interface)
{
	ASSERT_ALWAYS(runInSimulation);
}

/*
 * to be run in its own thread; will run forever
 *
 * poll for new trajectories and send parts of them to a robot as appropriate
 */
void freeFlyingCameraContinuousTrajSender::run()
{
	while(true)
	{
		trajsMux.lock();
		if(!trajs.empty())
		{
			/*
			 * remove next traj to run from list
			 */
			const trajspec traj = trajs[0];
			trajs.pop_front();
			trajsMux.unlock();

			bool success = true;
			const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();

#if 0 //not needed if movement isn't noisy
			/*
			 * ensure traj starts near where we are
			 */
			for(size_t i = 0; i < curConfig.size(); i++)
				if(fabs(curConfig[i] - traj.waypts[0][i]) > 1e-3/* TODO ? */)
				{
					cout << "[exec] ERROR: dof " << i << " of trajectory to use starts at " << traj.waypts[0][i] << " but robot is at " << curConfig[i] << endl;
					success = false;
					break;
				}
#endif

			if(success)
			{

			{
				std::lock_guard<std::mutex> lock(curGoalMux);
				curTrajEndConfiguration = traj.waypts.back();
			}

#if 0 //not sure how I want to do this
				/*
				 * add a waypt for current joints to the front of the traj (TODO better way to do this?)
				 */
				std::vector<grasp_utils::JointPose7D> newWaypts(traj.waypts.size() + 1);
				std::copy(traj.waypts.begin(), traj.waypts.end(), newWaypts.begin());
				newWaypts.back() = current config;
				traj.waypts = std::move(newWaypts);
#endif

				cout << "[exec] sending trajectory of length " << traj.waypts.size() << " to robot" << endl;
				cout << "[exec] traj starts at "; std::copy(traj.waypts[0].begin(), traj.waypts[0].end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
				cout << "[exec] traj ends at "; std::copy(traj.waypts.back().begin(), traj.waypts.back().end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
				cout << "[exec] robot starts at "; std::copy(curConfig.begin(), curConfig.end(), std::ostream_iterator<double>(cout, " ")); cout << endl;

				if(runInSimulation)
				{
					OpenRAVE::RobotBasePtr robot;
					OpenRAVE::RobotBase::ManipulatorPtr manip;
				{
					raveEnvLock envLock(simulationEnv, "simulationEnv in trajSender");
					std::vector<OpenRAVE::RobotBasePtr> robots;
					simulationEnv->GetRobots(robots);
					ASSERT_ALWAYS(robots.size() >= 1);
					robot = robots[0];
					manip = robot->GetActiveManipulator();
				}
					for(size_t i = 0; i < traj.waypts.size(); i++)
					{
					//	cout << "[exec] sleeping for " << traj.deltaT << endl;
						std::this_thread::sleep_for(std::chrono::milliseconds(uint32_t(traj.deltaT * 1e3)));

						std::vector<OpenRAVE::dReal> dofs(traj.waypts[i].size());
						std::copy(traj.waypts[i].begin(), traj.waypts[i].end(), dofs.begin());
					{
						raveEnvLock envLock(simulationEnv, "simulationEnv in trajSender");
						robotInterface->setRAVERobotConfiguration(robot, dofs);
					}

					}
				}
				else
				{
					ASSERT_ALWAYS(false);
				}

			{
				const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();
				cout << "[exec] robot ends at "; std::copy(curConfig.begin(), curConfig.end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
#if 0 //not needed if movement isn't noisy
				if(success)
				{
					for(size_t i = 0; i < curConfig.size(); i++)
						if(fabs(curConfig[i] - traj.waypts.back()[i]) > 1e-3/* TODO ? */)
						{
							cout << "[exec] ERROR: dof " << i << " of trajectory to use ends at " << traj.waypts.back()[i] << " but robot is at " << curConfig[i] << endl;
							success = false;
							break;
						}
				}
#endif
			}

			}

			trajCompletionSignal(traj.id, success);
		}
		else
		{
			trajsMux.unlock();
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}
}
