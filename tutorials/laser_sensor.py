# import statements
from datetime import datetime, timedelta
import numpy as np
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
import plotly.graph_objects as go
from LaserMeasurementModel import LaserMeasurementModel
from stonesoup.types.detection import Detection
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track
from stonesoup.functions import cart2pol, pol2cart
from stonesoup.types.array import StateVector
from math import gamma
from stonesoup.types.update import GaussianStateUpdate


def extract_dt_signal(path):
    """
    puts a "path" objects into a matrix for easier processing later
    :param path: Stone Soup object holding discrete-time vector signal (possibly measurements, control inputs, ground
    truth states, estimated states, etc.)
    :return: a (N, n) numpy matrix where "N" is the number of samples, "n" is the number of variables at each sample
    """
    if len(path) == 0:
        # return if there is no state variable to extract
        return

    if type(path) == type([]):
        ret = np.zeros((len(path), len(path[0].state_vector)))
        for k in range(0, len(path)):
            for i in range(0, len(path[0].state_vector)):
                ret[k, i] = path[k].state_vector[i]
        return ret
    else:
        ret = np.zeros((len(path), len(path.state.state_vector)))
        for k in range(0, len(path)):
            for i in range(0, len(path.state.state_vector)):
                ret[k, i] = path[k].state_vector[i]
        return ret


def plot_3d_path(dt_signal, pos_indices):
    """
    returns a figure object depicting the 3D path traced by a discrete-time vector signal containing (at least) <x,y,z>
    coordinates at each time step (as represented by a numpy matrix)
    :param dt_signal: a Nxn numpy matrix, where "N" is the number of time steps and "n" is the number of signals in the
    vector signal; represents the discrete-time signal containing (at least) the <x,y,z> coordinates at each time step
    :param pos_indices: a native python "list" object whose elements are the indices of the position coordinates in the
    "dt_signal" (must be size 3)
    :return: a Figure object from "plotly" library

    Usage:
    my_dt_signal = ...
    my_fig = plot_3d_path(my_dt_signal, [0,2,4])  # 0,2,4 are the indices of "my_dt_signal" containing the <x,y,z> coordinates respectively
    my_fig.show()
    """
    if len(pos_indices) != 3:
        # if the number of position indices that we receive is not strictly 3, then there was a mistake using the
        # function and just return
        return

    x = dt_signal[:, pos_indices[0]]
    y = dt_signal[:, pos_indices[1]]
    z = dt_signal[:, pos_indices[2]]
    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        marker=dict(size=4, color='blue'),
        line=dict(color='darkblue', width=2)
    ))

    fig.update_layout(width=800, height=700, autosize=False)
    return fig


def spherical_to_cartesian(dt_signal):
    # ordering (theta, phi, r)
    elevation = dt_signal[:, 0]  # theta
    bearing = dt_signal[:, 1]  # phi (aka azimuth)
    range = dt_signal[:, 2]  # r (range)
    ret = np.zeros(dt_signal.shape)
    for k in np.arange(0, ret.shape[0], 1):
        ret[k, 0] = range[k] * np.cos(elevation[k]) * np.cos(bearing[k])
        ret[k, 1] = range[k] * np.cos(elevation[k]) * np.sin(bearing[k])
        ret[k, 2] = range[k] * np.sin(elevation[k])
    return ret


def cyln_to_cart(dt_signal):
    # ordering (x,xdot,y,ydot,z,zdot)
    r = dt_signal[:, 0]
    rdot = dt_signal[:, 1]
    theta = dt_signal[:, 2]
    thetadot = dt_signal[:, 3]
    z = dt_signal[:, 4]
    zdot = dt_signal[:, 5]

    ret = np.zeros(dt_signal.shape)
    for k in range(0, ret.shape[0]):
        ret[k, 0] = r[k] * np.cos(theta[k])
        ret[k, 1] = rdot[k] * np.cos(theta[k]) - r[k] * np.sin(theta[k]) * thetadot[k]
        ret[k, 2] = r[k] * np.sin(theta[k])
        ret[k, 3] = rdot[k] * np.sin(theta[k]) + r[k] * np.cos(theta[k]) * thetadot[k]
        ret[k, 4] = z[k]
        ret[k, 5] = zdot[k]
    return ret


class LaserReceiver:
    def __init__(self, x, y, z, fov):
        self.x = x
        self.y = y
        self.z = z
        self.fov = fov  # must be defined in radians!
        self.raw_measurements = []

    def is_outside_fov(self, elevation):
        # Note: elevation is up and down. bearing is side-to-side
        # Note: elevation = -90 is "downward" so 0 is along like x+ axis
        if elevation > self.fov / 2.0 - np.pi / 2.0:
            return True
        return False

    def clear(self):
        self.raw_measurements = []

    def append_detection(self, measurement_no_noise, measurement):
        """
        appends a measurement to the measurement list if the measurement is within the FOV of the receiver
        :param measurement:
        :return:
        """
        elevation = measurement_no_noise[0]  # how do we know the measurement input argument will have this form?
        if not laser_receiver.is_outside_fov(elevation):
            self.raw_measurements.append(
                Detection(measurement, timestamp=state.timestamp, measurement_model=laser_measurement_model)
            )

    def get_detections_cartesian(self):
        """
        returns a Nxp numpy matrix, where "N" is the number of measurements and "n" is the number of measurement
        variables (in this particular case, p = 3 because we measure the elevation, bearing, and range to the target)
        :return:
        """
        measurements_dt_signal_spherical = extract_dt_signal(self.raw_measurements)
        measurements_dt_signal_cartesian = spherical_to_cartesian(measurements_dt_signal_spherical)
        # put these cartesian coordinates in the space-frame
        for k in range(0, measurements_dt_signal_cartesian.shape[0]):
            measurements_dt_signal_cartesian[k, 0] = measurements_dt_signal_cartesian[k, 0] + self.x
            measurements_dt_signal_cartesian[k, 1] = measurements_dt_signal_cartesian[k, 1] + self.y
            measurements_dt_signal_cartesian[k, 2] = measurements_dt_signal_cartesian[k, 2] + self.z
        return measurements_dt_signal_cartesian

# we start check the current datetime, so we can measure elapsed time by end of simulation
start_time = datetime.now()

# Set rng seed to 1991 to probe this particular example repeatedly
np.random.seed(1991)

# These next 5 lines (including comments) create a model defining the dynamics of a moving target
q_r = 0.05  # covariance matrix constant for the "Nearly constant velocity" model in r-direction
q_theta = 0.05  # covariance matrix constant for the "Nearly constant velocity" model in theta-direction
q_z = 0.05  # covariance matrix constant for the "Nearly constant velocity" model in z-direction
# creates a model object following the form: x[k+1] = A x[k] + w[k], w[k] ~ Normal(0, Q), where
# A = [1 T; 0 1] and Q = [1/3*T^3 1/2*T^2; 1/2*T^2 T] where T is the sample time
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_r), ConstantVelocity(q_theta), ConstantVelocity(q_z)])

# The next 9 lines create and initialize a "truth path" object
truth = GroundTruthPath([GroundTruthState([0, 0.1, 0, 0, -0.5, 0], timestamp=start_time)])  # inital conditions are [0; 1; 0; 1]
num_steps = 30  # integrate over the next "num_steps" time steps
T = 1
for k in range(1, num_steps + 1):
    truth.append(
        GroundTruthState(
            transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=T)),
            timestamp=start_time+timedelta(seconds=k)
        )
    )

    truth.states[k].state_vector[4] = -0.5
    truth.states[k].state_vector[5] = 0

    # Make sure we stay inside the swimming envelope of [-5,0] (ft)
    # if truth.states[k].state_vector[4] > 0:
    #     truth.states[k].state_vector[4] = 0
    #     truth.states[k].state_vector[5] = -truth.states[k].state_vector[5]
    #     truth.states[k-1].state_vector[5] = (truth.states[k].state_vector[4] - truth.states[k-1].state_vector[4]) / T  # ensures consistency across history of speeds
    # if truth.states[k].state_vector[4] < -5:
    #     truth.states[k].state_vector[4] = -5
    #     truth.states[k].state_vector[5] = (truth.states[k].state_vector[4] - truth.states[k-1].state_vector[4]) / T  # ensures consistency across history of speeds

truth_dt_signal_cyl = extract_dt_signal(truth)
truth_dt_signal_cart = cyln_to_cart(truth_dt_signal_cyl)

# place the sensor above the area where we expect to find the target
laser_receiver = LaserReceiver(0, 0, 10, np.deg2rad(90))

fig = plot_3d_path(truth_dt_signal_cart, [0, 2, 4])
num_lines = 100
for i in range(0, num_lines):
    phi = (2 * np.pi - 0) / num_lines * i  # azimuth
    theta = -np.pi / 2 + laser_receiver.fov / 2  # elevation
    z = np.min(truth_dt_signal_cart[:, 4])  # extend down to the minimum z component of the ground truth path
    r = (z - laser_receiver.z) / np.sin(theta)
    fig.add_trace(go.Scatter3d(
        x=[laser_receiver.x, r * np.cos(theta) * np.cos(phi)], y=[laser_receiver.y, r * np.cos(theta) * np.sin(phi)], z=[laser_receiver.z, z],
        mode="lines",
        line=dict(
            color='darkblue',
            width=2
        ),
        opacity=0.5,
        showlegend=False
    ))
fig.show()

R = np.diag([0.1, 0.1, 0.1])
laser_measurement_model = LaserMeasurementModel(
    ndim_state=6,
    mapping=(0, 2, 4),
    noise_covar=R,  # covariance matrix
    translation_offset=np.array([[laser_receiver.x], [laser_receiver.y], [laser_receiver.z]]),  # offset measurements to location of sensor in cartesian.
    laser_power=4  # looked up and 4000 mWatts is pretty good
)

ebr_measurement_model = CartesianToElevationBearingRange(
    ndim_state=6,
    mapping=(0, 2, 4),
    noise_covar=np.diag([np.radians(0.01), np.radians(0.01), 0.01]),  # Covariance matrix. 0.2 degree variance in bearing and 1 metre in range
    translation_offset=np.array([[laser_receiver.x], [laser_receiver.y], [laser_receiver.z]])  # Offset measurements to location of sensor in cartesian.
)

measurements = []
for state in truth:
    # measurement_no_noise = measurement_model2.function(state, noise=False)  # make sure we only add noise if it's outside FOV, not just remove entirely...
    measurement = laser_measurement_model.function(state, noise=True)
    measurements.append(
        Detection(measurement, timestamp=state.timestamp, measurement_model=laser_measurement_model)
    )
    # laser_receiver.append_detection(measurement_no_noise, measurement)
#
predictor = ExtendedKalmanPredictor(transition_model)  # we only want to add a subset of transition_model
prior = GaussianState([[0, 0.1, 0, 0, -0.5, 0]], np.diag([1.5, 0.5, 1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

#
track = Track()
for measurement in measurements:
    # 1. Prediction Phase
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)
    xhat_k_k1 = prediction.state_vector
    P_k_k1 = prediction.covar

    # 2. "Residual Calculation" Phase
    y_tilde = measurement.state_vector - laser_measurement_model.function(prediction)

    # Compute "C" matrix on-the-fly
    n = laser_measurement_model.beam_order
    P = laser_measurement_model.laser_power
    Theta = laser_measurement_model.divergence
    r = xhat_k_k1[0]
    rdot = xhat_k_k1[1]
    theta = xhat_k_k1[2]
    thetadot = xhat_k_k1[3]
    z = xhat_k_k1[4]
    zdot = xhat_k_k1[5]
    sigma = np.power(1/10, -1/n) * z * np.sin(Theta / 2)
    c = (-np.square(n) * np.power(r, n-1) * P) / (np.power(4, 1/n + 1) * np.pi * np.power(sigma, n+2) * gamma(2/n))
    Ck = np.array([
        [c * np.exp(-1/2 * np.power(r/sigma, n)), 0, 0, 0],
        [-thetadot * np.sin(theta), -rdot * np.sin(theta) - r * thetadot * np.cos(theta), np.cos(theta), -r * np.sin(theta)],
        [thetadot * np.cos(theta), rdot * np.cos(theta) - r * thetadot * np.sin(theta), np.sin(theta), r * np.cos(theta)],
    ])


    Sk = np.matmul(np.matmul(Ck, P_k_k1[:4, :4]), Ck.transpose()) + R  # the P_k_k1[:4,:4] ignores all z and zdots

    # 3. Update step
    K_k = np.matmul(np.matmul(P_k_k1[:4, :4], Ck.transpose()), np.linalg.inv(Sk))
    xhat_k_k = xhat_k_k1[:4] + np.matmul(K_k, y_tilde)
    P_k_k = np.matmul((np.eye(4) - np.matmul(K_k, Ck)), P_k_k1[:4,:4])

    # stonesoup book keeping
    xhat_k_k_extended = np.concatenate([xhat_k_k.base, np.zeros((2, 1))])
    xhat_k_k_extended[4] = xhat_k_k1[4]  # maybe just repeat don't predict?
    xhat_k_k_extended[5] = xhat_k_k1[5]  # maybe just repeat don't predict?
    P_k_k_extended = np.zeros((6, 6))
    P_k_k_extended[:4, :4] = P_k_k
    P_k_k_extended[4:, 4:] = P_k_k1[4:, 4:]  # maybe just repeat, don't predit?
    post = GaussianStateUpdate(state_vector=xhat_k_k_extended, covar=P_k_k_extended, hypothesis=hypothesis, timestamp=measurement.timestamp)
    track.append(post)
    prior = track[-1]


track_dt_signal = extract_dt_signal(track)
track_dt_signal_cart = cyln_to_cart(track_dt_signal)
fig3 = plot_3d_path(track_dt_signal_cart, [0, 2, 4])
fig3.show()

from plotly.subplots import make_subplots
error_signal = track_dt_signal_cart - truth_dt_signal_cart
histograms = make_subplots(rows=2, cols=3)
histograms.add_trace(
    go.Histogram(x=error_signal[:,0]),
    row=1, col=1
)

histograms.add_trace(
    go.Histogram(x=error_signal[:,1]),
    row=2, col=1
)

histograms.add_trace(
    go.Histogram(x=error_signal[:,2]),
    row=1, col=2
)

histograms.add_trace(
    go.Histogram(x=error_signal[:,3]),
    row=2, col=2
)

histograms.add_trace(
    go.Histogram(x=error_signal[:,4]),
    row=1, col=3
)

histograms.add_trace(
    go.Histogram(x=error_signal[:,5]),
    row=2, col=3
)

histograms.update_layout(height=600, width=800, title_text="Error Signal Histograms (est - act)")
histograms.show()


