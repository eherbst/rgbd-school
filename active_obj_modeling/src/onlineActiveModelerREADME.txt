Evan Herbst
description of the onlineActiveModeler class for Uber
6 / 1 / 15
------------------

onlineActiveModeler is part of a project to do online 1) scene modeling, 2) change detection, 3) object modeling, 4) view planning all in one. This class builds on 1), 2) and 3) and adds 4). At a high level, the class suggests next poses to view the scene from, scores them according to a utility function, picks the best-scoring one, plans a configuration-space path to it, and sends the path to a robot controller. I implemented controllers for a Barrett WAM holding a camera, a Baxter holding a camera, and a virtual free-flying camera. The class is used something like

--------------------------------------------------------

onlineActiveModeler modeler;

thread 1
-------------
while(new RGBD frame available && !modeler.isExperimentOver())
   modeler.updatePerModelingFrame();
   visualize modeler.getTargetCamPoseImg(), modeler.getMeshValuesImg(), etc.

thread 2
-------------
while(1)
   modeler.updateAsync();

--------------------------------------------------------

updatePerModelingFrame(), which is called synchronously with receiving new frames, currently only saves "scene differencing" (change detection) information for the map with respect to some previous map. We use this for some types of view scoring.

updateAsync() is more interesting:

At each frame, we suggest next camera poses (technically not 'next', since we're trying to plan several seconds out while we're moving toward a previously planned pose). This is done from initCamPoseSuggestion() and suggestMoreCamPoses(): we suggest only a small batch of poses at once so that, should one of the first few prove good, we don't have to spend exorbitant amounts of time suggesting millions of poses around the entire map. The suggest* functions are the inner suggestion loop. The map being operated on is volumetric, so we can easily get a triangulated mesh from it with marching cubes. This is where the omnipresent triangles in this code come from.

Each suggested pose is checked for collisions with the known environment, including surfaces at "frontiers" where known-free and yet-unseen space meet. We cull poses that are similar to poses we've already visited. We then render the map from each pose with openGL, writing a utility score for each mesh triangle into the color channel to make use of extremely fast hardware rendering, and sum over visible triangles to get a utility score per suggested pose. For cuboid maps ten feet across, we can process 500 - 5000 views a second even without custom shaders. The set of utility functions implemented is given by the triValueFuncType enum. We use such utilities as whether we've seen the area yet and whether there seems to be a movable object that we could model more closely.

For poses in order by utility, we then try to plan a path (using the plan*Path* functions) from the plan start configuration (where we expect to be a few seconds from now) to each suggested one. At the first successful plan (represented as a motionPlanInfo struct), we command the robot to follow the path. I have functions to use simple linear interpolation of configurations, an RRT, or CHOMP, a gradient-descent method (faster but less reliable than RRTs, and prefers volumetric maps), for planning. Checking how close the hardware got to the goal, by doing some sort of visual matching after the robot claims to have finished moving, is future work, as is interrupting a currently queued motion once its target is no longer as interesting as another view.

(Fast) mapping continues while this (slow) planning is running, so at the beginning of each planning iteration, we capture all the map info we need, under synchronization, in processMapsBeforeViewSelection(). This function is slow enough that adding a lock just for it is preferable to running it during the synchronous update, which happens much more often.

