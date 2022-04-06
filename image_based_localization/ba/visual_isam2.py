import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from gtsam.symbol_shorthand import L, X


def visual_ISAM2_plot(result):
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
        gtsam_plot.plot_pose3(fignum, pose_i, 10)
        i += 1

    # draw
    axes.set_xlim3d(-40, 40)
    axes.set_ylim3d(-40, 40)
    axes.set_zlim3d(-40, 40)
    plt.pause(1)

class VisualISAM2():
    def __init__(self):
        self.K = gtsam.Cal3_S2(50.0, 50.0, 0.0, 50.0, 50.0) # camera matrix
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0) # observation noise model

        # params
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.01)
        parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(parameters)

        # graph and values
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

        # frame id
        self.frame_id = 0

    def update(self, nearest_neighbor_pose, features):
        # add landmarks
        for landmark_id, landmark_position, measurement in features:
            self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
                measurement, self.measurement_noise, X(self.frame_id), L(landmark_id), self.K
            ))
            self.initial_estimate.insert(L(landmark_id), landmark_position)

        # use pose of nearest neighbor in map as initial estimate
        self.initial_estimate.insert(X(self.frame_id), nearest_neighbor_pose)

        if self.frame_id == 0:
            # add prior (30cm std on x,y,z 0.1 rad on roll,pitch,yaw)
            pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]
            ))
            self.graph.push_back(gtsam.PriorFactorPose3(X(0), nearest_neighbor_pose, pose_noise))
            point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
            self.graph.push_back(gtsam.PriorFactorPoint3(
                L(0), gtsam.Point3(0,0,0), point_noise))
        else:
            # optimize
            self.isam.update(self.graph, self.initial_estimate)
            self.current_estimate = self.isam.calculateEstimate()
        self.frame_id += 1

    def plot(self):
        visual_ISAM2_plot(self.current_estimate)
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    visam2 = VisualISAM2()
