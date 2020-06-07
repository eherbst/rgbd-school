/*
 * testCHOMP: run chomp motion planning
 *
 * Evan Herbst
 * 11 / 1 / 13
 */

#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <openrave-core.h>
#include "rgbd_util/timer.h"
#include "rgbd_util/assert.h"
#include "openrave_utils/openraveUtils.h"
#include "orcdchomp/orcdchomp_mod.h"
using std::vector;
using std::string;
using std::ostream;
using std::istringstream;
using std::ostringstream;
using std::stringstream;
using std::cout;
using std::endl;

int main(int argc, char* argv[])
{
	configOptions cfg;
	//for openraveUtils
	cfg.set("raveBasePath", "/home/eherbst/proj/ros-extras/openrave_planning/openrave/share/openrave-latest");

	OpenRAVE::RaveInitialize(true);
	OpenRAVE::RaveSetDebugLevel(OpenRAVE::Level_Debug);
	OpenRAVE::EnvironmentBasePtr env = initOpenRAVEwithRobot(cfg);
	OpenRAVE::RobotBasePtr robot = env->GetRobot("BarrettWAM");
	ASSERT_ALWAYS(robot);
	OpenRAVE::RobotBase::ManipulatorPtr manip = robot->GetActiveManipulator();
	const std::vector<int> activeDOFIndices = manip->GetArmIndices();

	/*
	 * load IKFast and generate a model for our robot
	 */
	OpenRAVE::ModuleBasePtr pikfast = OpenRAVE::RaveCreateModule(env, "ikfast");
	env->AddModule(pikfast, ""/* cmd-line args */);
	stringstream ssin, ssout;
	ssin << "LoadIKFastSolver " << robot->GetName() << " " << (int)OpenRAVE::IKP_Transform6D;
	if( !pikfast->SendCommand(ssout,ssin) ) {
		cout << "failed to load iksolver" << endl;
		return 1;
	}
	if( !manip->GetIkSolver()) {
		cout << "no ik solver; dying" << endl;
		env->Destroy();
		return 1;
	}

	OpenRAVE::ModuleBasePtr chompModule = OpenRAVE::RaveCreateModule(env, "orcdchomp");
	ASSERT_ALWAYS(chompModule);
	env->AddModule(chompModule, ""/* cmd-line args */);

	robot->SetActiveDOFs(manip->GetArmIndices());
	vector<OpenRAVE::dReal> vlower,vupper;
	robot->GetActiveDOFLimits(vlower,vupper);
	while(true)
	{
		OpenRAVE::EnvironmentMutex::scoped_lock lock(env->GetMutex()); // lock environment

		// move robot randomly
		vector<OpenRAVE::dReal> v(manip->GetArmIndices().size());
		for(size_t i = 0; i < vlower.size(); ++i) v[i] = vlower[i] + (vupper[i]-vlower[i]) * OpenRAVE::RaveRandomFloat();
		robot->SetActiveDOFValues(v);
		const bool bincollision = env->CheckCollision(robot) || robot->CheckSelfCollision();
		if(bincollision) cout << "got collision at ik target pose" << endl;
		else
		{
			/*
			 * if the target pose is feasible, run IK solving
			 */
			rgbd::timer t;
			vector<OpenRAVE::dReal> soln;
			const bool success = manip->FindIKSolution(manip->GetIkParameterization(OpenRAVE::IKP_Transform6D), soln, OpenRAVE::IKFO_CheckEnvCollisions);
			t.stop("run ik");
//			vector<vector<OpenRAVE::dReal>> solns;
//			const bool success = manip->FindIKSolutions(manip->GetIkParameterization(OpenRAVE::IKP_Transform6D), solns, OpenRAVE::IKFO_CheckEnvCollisions);
//			cout << "success " << success << "; got " << solns.size() << " solns" << endl;

			/*
			 * if IK solving succeeded, run motion planning
			 */
			if(success)
			{
				//cout << "ik target= " << v << ", soln= " << soln << endl;
				t.restart();

				/*
				 * do ik to get target joint values
				 */
				std::vector<double> targetJointAngles(activeDOFIndices.size());
			{
				vector<OpenRAVE::dReal> soln;
				const bool success = manip->FindIKSolution(manip->GetIkParameterization(OpenRAVE::IKP_Transform6D), soln, OpenRAVE::IKFO_CheckEnvCollisions);
				ASSERT_ALWAYS(success);
				std::copy(soln.begin(), soln.end(), targetJointAngles.begin());

				robot->SetActiveDOFValues(soln); //for visualization
				cout << "displaying target end effector pose; hit enter to run motion planning" << endl;
				int qq; std::cin >> qq;
				vector<OpenRAVE::dReal> curDOFs(activeDOFIndices.size(), 0);
				robot->SetActiveDOFValues(curDOFs);
			}

				bool planningSuccess = true;
				std::ostringstream outstr;
			{
				std::stringstream args;

				std::vector<OpenRAVE::KinBodyPtr> kinbodies;
				env->GetBodies(kinbodies);
				for(const auto& b : kinbodies)
					if(b->GetName() != "BarrettWAM")
					{
						//when using computedistancefield, all bodies other than the one being computed for must be disabled
						for(const auto& b2 : kinbodies)
							if(b2 != b)
								b2->Enable(false);

						cout << "computingdistancefield " << b->GetName() << endl;
						args << "computedistancefield kinbody " << b->GetName();
						if(!chompModule->SendCommand(outstr, args))
						{
							cout << "computedistancefield '" << b->GetName() << "' failed" << endl;
							ASSERT_ALWAYS(false);
						}

						for(const auto& b2 : kinbodies)
							if(b2 != b)
								b2->Enable(true);
					}
			}
				outstr.str("");
			{
				std::stringstream args;
				args << "create robot " << "BarrettWAM" << " adofgoal " << "\"";
				std::copy(targetJointAngles.begin(), targetJointAngles.end(), std::ostream_iterator<double>(args, " "));
				args << "\"" << " lambda 200 n_points 100 epsilon .1";
				if(!chompModule->SendCommand(outstr, args))
				{
					ASSERT_ALWAYS(false && "create failed");
				}
			}
				const std::string runptrstr = outstr.str();
				outstr.str("");
			{
				std::stringstream args;
				args << "iterate run " << runptrstr << " n_iter 150";
				if(!chompModule->SendCommand(outstr, args))
				{
					ASSERT_ALWAYS(false && "iterate failed");
				}
			}
				outstr.str("");
			{
				std::stringstream args;
				args << "gettraj run " << runptrstr << " retime_trajectory";
				try
				{
					if(!chompModule->SendCommand(outstr, args))
					{
						ASSERT_ALWAYS(false && "gettraj failed");
					}
				}
				catch(const OpenRAVE::openrave_exception& x) //probably a collision with the map during the returned trajectory
				{
					cout << "exception in motion planning: " << x.what() << endl;
					planningSuccess = false;
				}
			}
				OpenRAVE::TrajectoryBasePtr traj = OpenRAVE::RaveCreateTrajectory(env, "");
				ASSERT_ALWAYS(traj); //EVH: we've customized orcdchomp so that gettraj returns the trajectory even if it throws a collision exception
				std::istringstream instr(outstr.str());
				traj->deserialize(instr);
				outstr.str("");
			{
				std::stringstream args;
				args << "destroy run " << runptrstr;
				if(!chompModule->SendCommand(outstr, args))
				{
					ASSERT_ALWAYS(false && "destroy failed");
				}
			}
				outstr.str("");

				std::vector<OpenRAVE::dReal> trajptvec;
				traj->GetWaypoints(0, traj->GetNumWaypoints(), trajptvec);
				cout << "traj:" << endl;
				const size_t waypointLength = 2 * activeDOFIndices.size() + 1; //2n + 1 because position, velocity, delta-time
				ASSERT_ALWAYS(waypointLength * traj->GetNumWaypoints() == trajptvec.size());
				for(size_t q = 0; q < traj->GetNumWaypoints(); q++)
				{
					for(int i : activeDOFIndices) cout << trajptvec[waypointLength * q + i] << ' ';
					cout << endl;
				}
				cout << "traj done" << endl;
				int qq; std::cin >> qq;

				t.stop("run motion planning");
			}
		}
	}

	OpenRAVE::RaveDestroy();
	return 0;
}

