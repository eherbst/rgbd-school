/*
 * marvinContinuousTrajSender: wrapper for trajectory executor to allow continual replanning
 *
 * Evan Herbst
 * 1 / 1 / 14
 */

#include <iostream>
#include <thread>
#include "grasp_utils/ExecuteGraspTest.h" //the service to run a trajectory
#include "active_vision_common/openraveUtils.h"
#include "active_obj_modeling/marvinContinuousTrajSender.h"
using std::cout;
using std::endl;

/*
 * if simulationEnv, we'll run everything in simulation
 */
marvinContinuousTrajSender::marvinContinuousTrajSender(const std::shared_ptr<cameraOnRobotHandler>& camHandler, const OpenRAVE::EnvironmentBasePtr& simulationEnv)
: continuousTrajSender(camHandler, simulationEnv)
{
	if(!runInSimulation)
	{
		nh_global.reset(new ros::NodeHandle);
		cout << "[exec] waiting for marvin execute-grasp service" << endl;
		runTrajectoryServiceClient = nh_global->serviceClient<grasp_utils::ExecuteGraspTest>("/execute_grasp_test");
		runTrajectoryServiceClient.waitForExistence();
	}
}

/*
 * to be run in its own thread; will run forever
 *
 * poll for new trajectories and send parts of them to a robot as appropriate
 */
void marvinContinuousTrajSender::run()
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

			/*
			 * ensure traj starts near where we are
			 */
			const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();
			ASSERT_ALWAYS(curConfig.size() == traj.waypts[0].size());
			for(size_t i = 0; i < curConfig.size(); i++)
				if(fabs(curConfig[i] - traj.waypts[0][i]) > 5/* degrees; TODO ? */ * M_PI / 180)
				{
					cout << "[exec] ERROR: dof " << i << " of trajectory to use starts at " << traj.waypts[0][i] << " but robot is at " << curConfig[i] << endl;
					success = false;
					break;
				}

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
						robot->SetActiveDOFs(manip->GetArmIndices());
						robot->SetActiveDOFValues(dofs);
					}

					}
				}
				else
				{
					grasp_utils::ExecuteGraspTestRequest request;
					request.params.trajectory.waypoints.resize(traj.waypts.size());
					for(size_t i = 0; i < traj.waypts.size(); i++) std::copy(traj.waypts[i].begin(), traj.waypts[i].end(), request.params.trajectory.waypoints[i].joints.begin());
					request.params.timestep = traj.deltaT;
					request.params.length = request.params.trajectory.waypoints.size();
					request.params.close_fingers = false;
					grasp_utils::ExecuteGraspTestResponse response;
					/*
					 * this call shouldn't finish until the movement has finished
					 */
					if(!runTrajectoryServiceClient.call(request, response))
					{
						cout << "[exec] ERROR: failed to call run-traj service " << endl;
						success = false;
					}
					else if(!response.done)
					{
						cout << "[exec] ERROR: trajectory didn't finish running" << endl;
						success = false;
					}
					else
					{
						cout << "[exec] trajectory finished running" << endl;
						success = true;
					}
				}

			{
				const std::vector<OpenRAVE::dReal> curConfig = robotCamHandler->getLatestConfiguration();
				cout << "[exec] robot ends at "; std::copy(curConfig.begin(), curConfig.end(), std::ostream_iterator<double>(cout, " ")); cout << endl;
				if(success)
				{
					for(size_t i = 0; i < curConfig.size(); i++)
						if(fabs(curConfig[i] - traj.waypts.back()[i]) > 3/* TODO ? */ * M_PI / 180)
						{
							cout << "[exec] ERROR: dof " << i << " of trajectory to use ends at " << traj.waypts.back()[i] << " but robot is at " << curConfig[i] << endl;
							success = false;
							break;
						}
				}
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
