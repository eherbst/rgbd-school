/*
 * baxterContinuousTrajSender: wrapper for trajectory executor to allow continual replanning
 *
 * Evan Herbst
 * 1 / 1 / 14
 */

#include <iostream>
#include <thread>
#include <control_msgs/FollowJointTrajectoryGoal.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include "active_vision_common/openraveUtils.h"
#include "active_obj_modeling/baxterContinuousTrajSender.h"
using std::cout;
using std::endl;

/*
 * if simulationEnv, we'll run everything in simulation
 */
baxterContinuousTrajSender::baxterContinuousTrajSender(const std::shared_ptr<cameraOnRobotHandler>& camHandler, const OpenRAVE::EnvironmentBasePtr& simulationEnv)
: continuousTrajSender(camHandler, simulationEnv), nh_global()
{
	if(!runInSimulation)
	{
		runTrajClient.reset(new actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>("/robot/limb/right/follow_joint_trajectory", true/* spin in own thread */));
		cout << "waiting for baxter action server" << endl;
		runTrajClient->waitForServer();
		//rospy.on_shutdown(runTrajClient.cancelAllGoals()) TODO figure out how to do in c++
	}
}

/*
 * to be run in its own thread; will run forever
 *
 * poll for new trajectories and send parts of them to a robot as appropriate
 */
void baxterContinuousTrajSender::run()
{
	control_msgs::FollowJointTrajectoryGoal trajGoal;
	trajGoal.trajectory.joint_names = {"right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2"}; //list of joints; TODO better way to get?

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
					for(size_t i = 0; i < traj.waypts.size(); i++)
					{
						trajectory_msgs::JointTrajectoryPoint waypt;
						waypt.positions = traj.waypts[i];
						waypt.time_from_start = ros::Duration((i + 1) * traj.deltaT); //this will give us a bit of time at the beginning to make up for not-quite-matching actual and expected positions
						trajGoal.trajectory.points.push_back(waypt);
					}
					trajGoal.trajectory.header.stamp = ros::Time::now();
					runTrajClient->sendGoal(trajGoal);
					const bool finished = runTrajClient->waitForResult(ros::Duration(traj.waypts.size() * traj.deltaT + 10)/* timeout; TODO ? */);
					if(!finished)
					{
						cout << "[exec] ERROR: trajectory didn't finish before timeout; might be an actual problem" << endl;
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
