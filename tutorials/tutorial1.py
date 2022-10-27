# import statements
from datetime import datetime, timedelta
import numpy as np
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.plotter import Plotterly
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.state import GaussianState
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.track import Track

# we start check the current datetime, so we can measure elapsed time by end of simulation
start_time = datetime.now()

# Set rng seed to 1991 to probe this particular example repeatedly
np.random.seed(1991)

# These next 5 lines (including comments) create a model defining the dynamics of a moving target
q_x = 0.05  # covariance matrix constant for the "Nearly constant velocity" model in x-direction
q_y = 0.05  # covariance matrix constant for the "Nearly constant velocity" model in y-direction
# creates a model object following the form: x[k+1] = A x[k] + w[k], w[k] ~ Normal(0, Q), where
# A = [1 T; 0 1] and Q = [1/3*T^3 1/2*T^2; 1/2*T^2 T] where T is the sample time
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(q_x), ConstantVelocity(q_y)])

# The next 9 lines create and initialize a "truth path" object
truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])  # inital conditions are [0; 1; 0; 1]
num_steps = 20  # integrate over the next 20 time steps
for k in range(1, num_steps + 1):
    truth.append(
        GroundTruthState(
            transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
            timestamp=start_time+timedelta(seconds=k)
        )
    )

plotter = Plotterly()
plotter.plot_ground_truths(truth, [0, 2])
plotter.fig.show()


# Check out what the A and Q matrices would be if the sampling time were T=1
transition_model.matrix(time_interval=timedelta(seconds=1))
transition_model.covar(time_interval=timedelta(seconds=1))

# Now try to measure the system
# y = [1 0 0 0 ; 0 0 1 0] x[k] + v[k], v[k] ~ Normal(0, R), R = [1 0; 0 1] * omega, omega is some constant.
measurement_model = LinearGaussian(
    ndim_state=4,  # Number of state dimensions (position and velocity in 2D)
    mapping=(0, 2),  # Mapping measurement vector index to state index
    noise_covar=np.array([[5, 0],  # Covariance matrix for Gaussian PDF
                          [0, 5]])
)

# Check out the H and R matrices
measurement_model.matrix()
measurement_model.covar()

# Generate the measurements
measurements = []
for state in truth:
    measurement = measurement_model.function(state, noise=True)
    measurements.append(
        Detection(measurement, timestamp=state.timestamp, measurement_model=measurement_model)
    )

plotter.plot_measurements(measurements, [0, 2])
plotter.fig.show()

# Setup Kalman filter
predictor = KalmanPredictor(transition_model)  # the "predictor" object was given the (A,B) model
updater = KalmanUpdater(measurement_model)  # the "updater" object was given the (C,D) model

# initialize the Kalman filter
prior = GaussianState([[0], [1], [0], [1]], np.diag([1.5, 0.5, 1.5, 0.5]), timestamp=start_time)

# iterate through all the measurements applying the Kalman filter
track = Track()
for measurement in measurements:
    prediction = predictor.predict(prior, timestamp=measurement.timestamp)
    hypothesis = SingleHypothesis(prediction, measurement)  # Group a prediction and measurement
    post = updater.update(hypothesis)
    track.append(post)
    prior = track[-1]

# Plot the result
plotter.plot_tracks(track, [0, 2], uncertainty=True)
plotter.fig.show()  # what is the meaning of the light green circles around each prediction?
