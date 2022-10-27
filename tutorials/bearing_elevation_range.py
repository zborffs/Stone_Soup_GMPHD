# import statements
from datetime import datetime, timedelta
import numpy as np
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
import plotly.graph_objects as go
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.types.detection import Detection
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track


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
        marker=dict(size=4, color=z,colorscale='Viridis',),
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

# we start check the current datetime, so we can measure elapsed time by end of simulation
start_time = datetime.now()

# Set rng seed to 1991 to probe this particular example repeatedly
np.random.seed(1991)

# These next 5 lines (including comments) create a model defining the dynamics of a moving target
q_x = 0.05  # covariance matrix constant for the "Nearly constant velocity" model in x-direction
q_y = 0.05  # covariance matrix constant for the "Nearly constant velocity" model in y-direction
q_z = 0.05  # covariance matrix constant for the "Nearly constant velocity" model in z-direction
# creates a model object following the form: x[k+1] = A x[k] + w[k], w[k] ~ Normal(0, Q), where
# A = [1 T; 0 1] and Q = [1/3*T^3 1/2*T^2; 1/2*T^2 T] where T is the sample time
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x), ConstantVelocity(q_y), ConstantVelocity(q_z)])

# The next 9 lines create and initialize a "truth path" object
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1, -0.5, 0], timestamp=start_time)])  # inital conditions are [0; 1; 0; 1]
num_steps = 30  # integrate over the next "num_steps" time steps
T = 1
for k in range(1, num_steps + 1):
    # go this way for the first "num_steps" steps
    truth.append(
        GroundTruthState(
            transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=T)),
            timestamp=start_time+timedelta(seconds=k)
        )
    )

    # Make sure we stay inside the swimming envelope of [-5,0] (ft)
    if truth.states[k].state_vector[4] > 0:
        truth.states[k].state_vector[4] = 0
        truth.states[k].state_vector[5] = -truth.states[k].state_vector[5]
        truth.states[k-1].state_vector[5] = (truth.states[k].state_vector[4] - truth.states[k-1].state_vector[4]) / T  # ensures consistency across history of speeds
    if truth.states[k].state_vector[4] < -5:
        truth.states[k].state_vector[4] = -5
        truth.states[k].state_vector[5] = (truth.states[k].state_vector[4] - truth.states[k-1].state_vector[4]) / T  # ensures consistency across history of speeds

truth_dt_signal = extract_dt_signal(truth)

fig = plot_3d_path(truth_dt_signal, [0, 2, 4])
fig.show()

# place the sensor above the area where we expect to find the target
sensor_x = 0
sensor_y = 0
sensor_z = 10

measurement_model = CartesianToElevationBearingRange(
    ndim_state=6,
    mapping=(0, 2, 4),
    noise_covar=np.diag([np.radians(0.01), np.radians(0.01), 0.01]),  # Covariance matrix. 0.2 degree variance in bearing and 1 metre in range
    translation_offset=np.array([[sensor_x], [sensor_y], [sensor_z]])  # Offset measurements to location of sensor in cartesian.
)

measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(
        Detection(measurement, timestamp=state.timestamp, measurement_model=measurement_model)
    )

measurements_dt_signal_spherical = extract_dt_signal(measurements)
measurements_dt_signal_cartesian = spherical_to_cartesian(measurements_dt_signal_spherical)

# put these cartesian coordinates in the space-frame
for k in range(0, measurements_dt_signal_cartesian.shape[0]):
    measurements_dt_signal_cartesian[k, 0] = measurements_dt_signal_cartesian[k, 0] + sensor_x
    measurements_dt_signal_cartesian[k, 1] = measurements_dt_signal_cartesian[k, 1] + sensor_y
    measurements_dt_signal_cartesian[k, 2] = measurements_dt_signal_cartesian[k, 2] + sensor_z

fig2 = plot_3d_path(measurements_dt_signal_cartesian, [0, 1, 2])
fig2.show()

#
predictor = ExtendedKalmanPredictor(transition_model)
updater = ExtendedKalmanUpdater(measurement_model)
prior = GaussianState([[0], [1], [0], [1], [-0.5], [0]], np.diag([1.5, 0.5, 1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

#
track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

track_dt_signal = extract_dt_signal(track)
fig3 = plot_3d_path(track_dt_signal, [0, 2, 4])
fig3.show()


from plotly.subplots import make_subplots
error_signal = track_dt_signal - truth_dt_signal
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