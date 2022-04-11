import gtsam
import gtsam.utils.plot as gtsam_plot
import numpy as np
from gtsam.symbol_shorthand import L, X
import matplotlib.pyplot as plt

def visual_ISAM2_plot(result, gt):
    """
    VisualISAMPlot plots current state of ISAM2 object
    Author: Ellon Paiva
    Based on MATLAB version by: Duy Nguyen Ta and Frank Dellaert
    """

    # Declare an id for the figure
    fignum = 0

    fig = plt.figure(fignum)
    axes = fig.gca(projection='3d')
    plt.cla()

    # Plot points
    # Can't use data because current frame might not see all points
    # marginals = Marginals(isam.getFactorsUnsafe(), isam.calculateEstimate())
    # gtsam.plot_3d_points(result, [], marginals)
    gtsam_plot.plot_3d_points(fignum, result, 'rx')

    # Plot cameras
    i = 0
    while result.exists(X(i)):
        pose_i = result.atPose3(X(i))
        if i==7:
            gtsam_plot.plot_pose3(fignum, pose_i, 0.6)
        else:
            gtsam_plot.plot_pose3(fignum, pose_i, 0.3)
        i += 1
    gtsam_plot.plot_pose3(fignum, gt, 1)

    # draw
    axes.set_xlim3d(-6, 6)
    axes.set_ylim3d(-6, 6)
    axes.set_zlim3d(-6, 6)
    plt.pause(2)

class VisualISAM2():
    def __init__(self):
        self.K = gtsam.Cal3_S2(585, 585, 0.0, 320, 240) # camera matrix
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1) # observation noise model
        self.point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.01)
        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.ones(6)*0.001
            # np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        )
        self.rot_body_2_camera = gtsam.Rot3(np.array([[0,-1,0],[0,0,-1],[1,0,0]]))

    def update(self, neighbor_ids, match_frame_ids, uv_points, xyz_points, pose_initial_guess, neighbor_poses, gt_pose):


        # graph and values
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        # plot
        plt.ion()
        fignum = 0
        fig = plt.figure(fignum)
        axes = fig.gca(projection='3d')
        plt.cla()
        axes.set_xlim3d(-1, 1)
        axes.set_ylim3d(-1, 1)
        axes.set_zlim3d(-1, 1)

        # update data to include test frame
        test_frame_id = len(neighbor_ids)
        id_map = {x:i for i, x in enumerate(neighbor_ids)}
        match_frame_ids = [[test_frame_id]+[id_map[y] for y in x] for x in match_frame_ids]
        neighbor_ids = neighbor_ids + [test_frame_id]
        neighbor_poses = neighbor_poses + [pose_initial_guess]

        # create projection factors
        
        for landmark_id, match_frame_id in enumerate(match_frame_ids):
            measurements = uv_points[landmark_id]
            landmark_position = xyz_points[landmark_id]
            initial_estimate.insert(L(landmark_id), gtsam.Point3(*landmark_position))
            graph.push_back(gtsam.PriorFactorPoint3(
                L(landmark_id), gtsam.Point3(*landmark_position), self.point_noise
            ))
            for measurement_index, frame_id in enumerate(match_frame_id):
                graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                    measurements[measurement_index], self.measurement_noise, X(frame_id),
                    L(landmark_id), self.K
                ))

        # gtsam_plot.plot_3d_points(fignum, initial_estimate, 'rx')
    
        # set neighbor initial poses
        for i, pose in enumerate(neighbor_poses):
            pose = gtsam.Pose3(
                gtsam.Rot3.Quaternion(pose[6],*pose[3:6]).inverse(),
                gtsam.Point3(pose[0:3])
            )
            # gtsam_plot.plot_pose3(fignum, pose, 0.1)
            initial_estimate.insert(X(i), pose)
            if i != test_frame_id:
                graph.push_back(gtsam.PriorFactorPose3(
                    X(i), pose, self.pose_noise
                ))

        # isam optimize
        # parameters = gtsam.ISAM2Params()
        # parameters.setRelinearizeThreshold(0.01)
        # isam = gtsam.ISAM2(parameters)
        # isam.update(graph, initial_estimate)
        # self.current_estimate = isam.calculateEstimate()

        # result
        gt_pose = gtsam.Pose3(
            gtsam.Rot3.Quaternion(gt_pose[6],*gt_pose[3:6]),
            gtsam.Point3(gt_pose[0:3])
        )
        initial_pose = gtsam.Pose3(
            gtsam.Rot3.Quaternion(pose_initial_guess[6],*pose_initial_guess[3:6]).inverse(),
            gtsam.Point3(pose_initial_guess[0:3])
        )

        # batch optimize
        optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate)
        try:
            self.current_estimate = optimizer.optimize()
        except:
            return initial_pose, gtsam.Pose3(), gt_pose

        # plot

        estimated_pose = self.current_estimate.atPose3(X(test_frame_id))
        estimated_pose = gtsam.Pose3(
            estimated_pose.rotation().inverse(),
            estimated_pose.translation()
        )

        gtsam_plot.plot_pose3(fignum, initial_pose, 0.5)
        gtsam_plot.plot_pose3(fignum, estimated_pose, 1)
        gtsam_plot.plot_pose3(fignum, gt_pose, 2)
        plt.close()

        return initial_pose, estimated_pose, gt_pose

    def update_smart_factor(self, neighbor_ids, match_frame_ids, uv_points, pose_initial_guess, neighbor_poses):

        plt.ion()

        # graph and values
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        # update data to include test frame
        test_frame_id = len(neighbor_ids)
        id_map = {x:i for i, x in enumerate(neighbor_ids)}
        match_frame_ids = [[test_frame_id]+[id_map[y] for y in x] for x in match_frame_ids]
        neighbor_ids = neighbor_ids + [test_frame_id]
        neighbor_poses = neighbor_poses + [pose_initial_guess]

        # create projection factors
        
        for landmark_id, match_frame_id in enumerate(match_frame_ids):
            measurements = uv_points[landmark_id]
            smart_factor = gtsam.SmartProjectionPose3Factor(self.measurement_noise, self.K)
            for measurement_index, frame_id in enumerate(match_frame_id):
                smart_factor.add(measurements[measurement_index], X(frame_id))
            graph.push_back(smart_factor)

        # set neighbor initial poses
        for i, pose in enumerate(neighbor_poses):
            pose = gtsam.Pose3(
                gtsam.Rot3.Quaternion(pose[6],*pose[3:6])
                    .compose(self.rot_body_2_camera),
                gtsam.Point3(pose[0:3])
            )
            initial_estimate.insert(X(i), pose)
            if i != test_frame_id:
                graph.push_back(gtsam.PriorFactorPose3(
                    X(i), pose, self.pose_noise
                ))

        # isam optimize
        # parameters = gtsam.ISAM2Params()
        # parameters.setRelinearizeThreshold(0.01)
        # isam = gtsam.ISAM2(parameters)
        # isam.update(graph, initial_estimate)
        # self.current_estimate = isam.calculateEstimate()

        # batch optimize
        optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate)
        self.current_estimate = optimizer.optimize()

        return self.current_estimate.atPose3(X(test_frame_id))
    

    def plot(self, gt):
        visual_ISAM2_plot(self.current_estimate, gt)
        # plt.ioff()
        # plt.show()


if __name__ == '__main__':
    visam2 = VisualISAM2()
