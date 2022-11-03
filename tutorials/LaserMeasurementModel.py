from abc import ABC
import copy
from typing import Sequence, Tuple, Union

import numpy as np
from scipy.linalg import inv, pinv, block_diag
from scipy.stats import multivariate_normal
from math import gamma
from stonesoup.types.array import Matrix

from stonesoup.base import Property, clearable_cached_property
from stonesoup.types.numeric import Probability

from stonesoup.functions import cart2pol, pol2cart, cart2sphere, sphere2cart, cart2angles, build_rotation_matrix
from stonesoup.types.array import StateVector, CovarianceMatrix, StateVectors
from stonesoup.types.angle import Bearing, Elevation
from stonesoup.models.measurement import MeasurementModel
from stonesoup.models.measurement.nonlinear import NonLinearGaussianMeasurement


class LaserMeasurementModel(NonLinearGaussianMeasurement):
    r"""This is a class implementation of a time-invariant measurement model, \
    where measurements are assumed to be received in the form of bearing \
    (:math:`\phi`), elevation (:math:`\theta`) and range (:math:`r`), with \
    Gaussian noise in each dimension.

    The model is described by the following equations:

    .. math::

      \vec{y}_t = h(\vec{x}_t, \vec{v}_t)

    where:

    * :math:`\vec{y}_t` is a measurement vector of the form:

    .. math::

      \vec{y}_t = \begin{bmatrix}
                \theta \\
                \phi \\
                r
            \end{bmatrix}

    * :math:`h` is a non-linear model function of the form:

    .. math::

      h(\vec{x}_t,\vec{v}_t) = \begin{bmatrix}
                asin(\mathcal{z}/\sqrt{\mathcal{x}^2 + \mathcal{y}^2 +\mathcal{z}^2}) \\
                atan2(\mathcal{y},\mathcal{x}) \\
                \sqrt{\mathcal{x}^2 + \mathcal{y}^2 + \mathcal{z}^2}
                \end{bmatrix} + \vec{v}_t

    * :math:`\vec{v}_t` is Gaussian distributed with covariance :math:`R`, i.e.:

    .. math::

      \vec{v}_t \sim \mathcal{N}(0,R)

    .. math::

      R = \begin{bmatrix}
            \sigma_{\theta}^2 & 0 & 0 \\
            0 & \sigma_{\phi}^2 & 0 \\
            0 & 0 & \sigma_{r}^2
            \end{bmatrix}

    The :py:attr:`mapping` property of the model is a 3 element vector, \
    whose first (i.e. :py:attr:`mapping[0]`), second (i.e. \
    :py:attr:`mapping[1]`) and third (i.e. :py:attr:`mapping[2]`) elements \
    contain the state index of the :math:`x`, :math:`y` and :math:`z`  \
    coordinates, respectively.

    Note
    ----
    The current implementation of this class assumes a 3D Cartesian plane.

    """  # noqa:E501

    translation_offset: StateVector = Property(
        default=None,
        doc="A 3x1 array specifying the Cartesian origin offset in terms of :math:`x,y,z` "
            "coordinates.")

    beam_order: int = Property(
        default=2,
        doc="The beam order. \"n\" in equations for intensity of beam"
    )

    laser_power: float = Property(
        default=1.0,
        doc="the laser's rated power. \"P\" in equations for intensity of beam"
    )

    divergence: float = Property(
        default=np.deg2rad(10),
        doc="the laser's divergence in radians. \"Theta\" in equations for intensity of beam"
    )

    def __init__(self, *args, **kwargs):
        """
        Ensure that the translation offset is initiated
        """
        super().__init__(*args, **kwargs)
        # Set values to defaults if not provided
        if self.translation_offset is None:
            self.translation_offset = StateVector([0] * 3)

    @property
    def ndim_meas(self) -> int:
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 4

    def function(self, state, noise=False, **kwargs) -> StateVector:
        r"""Model function :math:`h(\vec{x}_t,\vec{v}_t)`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_state`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        # what states do we need?
        # - 'x' or position of target along +x axis in fixed space-frame
        # - 'y' or position of target along +y axis in fixed space-frame
        # - 'z' or vertical distance along laser axis

        # Account for origin offset
        xyz = state.state_vector[self.mapping, :] - self.translation_offset  # compute (x-x0, y-y0, z-z0)

        # Rotate coordinates
        xyz_rot = self.rotation_matrix @ xyz  # perform rotation (not really necessary bc we wont use this)

        # extract the variables once more
        x = xyz_rot[0, :]
        y = xyz_rot[1, :]
        z = xyz_rot[2, :]

        # compute 'r'
        r = np.sqrt(np.square(x) + np.square(y))

        # use 'r' and 'z' to compute intensity
        sigma = np.power(1 / 10, -1 / self.beam_order) * z * np.sin(self.divergence / 2)
        I0 = (self.beam_order * self.laser_power) / (
                    2 * np.pi * (np.power(4, 1 / self.beam_order)) * np.square(sigma) * gamma(2 / self.beam_order))
        I = I0 * np.exp(-1 / 2 * np.power(r / sigma, self.beam_order))

        return StateVectors([I, Matrix([0.]), Matrix([0.]), Matrix([0.])]) + noise  # 0 b/c for now we are staying still

    def rvs(self, num_samples=1, **kwargs) -> Union[StateVector, StateVectors]:
        out = super().rvs(num_samples, **kwargs)
        out = np.array([[0.], [0.], [0.], [0.]]) + out
        return out
