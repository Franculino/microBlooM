from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import sys


class DistensibilityLawInitialise(ABC):
    """
    Abstract base class for initialiasing the distensibility law
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of DistensibilityLawInitialisation.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
        """
        Specify the reference pressures and diameters for each vessel
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class DistensibilityInitialiseNothing(DistensibilityLawInitialise):
    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
        """
        Do not update any diameters based on vessel distensibility
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class DistensibilityLawPassiveReferenceBaselinePressure(DistensibilityLawInitialise):
    """
    Define the reference state based on the current baseline values
    """

    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
        """
        Specify the reference pressure and diameter to the current baseline values (at time of initialisation)
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        distensibility.pressure_ref = np.copy(flownetwork.pressure)  # External pressure corresponds to baseline pres
        # Reference diameter corresponds to baseline diameter
        distensibility.diameter_ref = np.copy(flownetwork.diameter)[distensibility.eid_vessel_distensibility]


class DistensibilityLawPassiveReferenceConstantExternalPressureSherwin(DistensibilityLawInitialise):
    """
    Define the reference state based on a non-linear p-A ralation proposed by Sherwin et al. (2003).
    """
    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
        """
        Set the reference pressure to the constant external pressure. Compute a reference diameter for each
        vessel based on the specified reference pressure and the baseline diameters & pressures.
        Based on a non-linear p-A ralation proposed by Sherwin et al. (2003).
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # External pressure
        distensibility.pressure_external = self._PARAMETERS["pressure_external"]
        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_dist = distensibility.eid_vessel_distensibility
        # Assign a constant external pressure for the entire vasculature
        distensibility.pressure_ref = np.ones(flownetwork.nr_of_vs)*distensibility.pressure_external

        # Compute the reference diameter based on the constant external pressure and the baseline diameters & pressures.
        # Reference diameter does not correspond to the baseline diameter.
        pressure_difference_vertex = flownetwork.pressure - distensibility.pressure_ref
        pressure_difference_edge = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[
            eids_dist]

        # Solve quadratic formula for diameter ref {-b + sqrt(b^2-4*a*c)} (other solution is invalid)
        kappa = 2 * distensibility.e_modulus * distensibility.wall_thickness / (
                    (pressure_difference_edge) * (1. - np.square(distensibility.nu)))
        diameter_ref = .5 * (-kappa + np.sqrt(np.square(kappa) + 4 * kappa * flownetwork.diameter[eids_dist]))

        if True in (diameter_ref < .5 * flownetwork.diameter[eids_dist]):
            sys.exit("Error: Suspicious small reference diameter (compared to baseline diameter) detected. ")

        distensibility.diameter_ref = diameter_ref


class DistensibilityLawPassiveReferenceConstantExternalPressureUrquiza(DistensibilityLawInitialise):
    """
    Define the reference state based on a non-linear p-A ralation proposed by Urquiza et al. (2006).
    """
    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
        """
        Set the reference pressure to the constant external pressure. Compute a reference diameter for each
        vessel based on the specified reference pressure and the baseline diameters & pressures.
        Based on a non-linear p-A relation proposed by Urquiza et al. (2006).
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # External pressure
        distensibility.pressure_external = self._PARAMETERS["pressure_external"]
        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_dist = distensibility.eid_vessel_distensibility
        # Assign a constant external pressure for the entire vasculature
        distensibility.pressure_ref = np.ones(flownetwork.nr_of_vs)*distensibility.pressure_external

        # Compute the reference diameter based on the constant external pressure and the baseline diameters & pressures.
        # Reference diameter does not correspond to the baseline diameter.
        pressure_difference_vertex = flownetwork.pressure - distensibility.pressure_ref
        pressure_difference_edge = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[
            eids_dist]

        # Solve quadratic formula for diameter ref {-b + sqrt(b^2-4*a*c)} (other solution is invalid)
        kappa = 2 * distensibility.e_modulus * distensibility.wall_thickness / pressure_difference_edge
        diameter_ref = .5 * (-kappa + np.sqrt(np.square(kappa) + 4 * kappa * flownetwork.diameter[eids_dist]))

        if True in (diameter_ref < .5 * flownetwork.diameter[eids_dist]):
            sys.exit("Error: Suspicious small reference diameter (compared to baseline diameter) detected. ")

        distensibility.diameter_ref = diameter_ref


class DistensibilityLawPassiveReferenceConstantExternalPressureRammos(DistensibilityLawInitialise):
    """
    Define the reference state based on a linear p-A ralation proposed by Rammos et al. (1998).
    """
    def initialise_distensibility_ref_state(self, distensibility, flownetwork):
        """
        Set the reference pressure to the constant external pressure. Compute a reference diameter for each
        vessel based on the specified reference pressure and the baseline diameters & pressures.
        Based on a linear p-A relation proposed by Rammos et al. (1998).
        :param distensibility: distensibility object
        :type distensibility: source.distensibility.Distensibility
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        # External pressure
        distensibility.pressure_external = self._PARAMETERS["pressure_external"]
        # Edge ids that are a vessel with distensibility (diameter changes are possible)
        eids_dist = distensibility.eid_vessel_distensibility
        # Assign a constant external pressure for the entire vasculature
        distensibility.pressure_ref = np.ones(flownetwork.nr_of_vs)*distensibility.pressure_external

        # Compute the reference diameter based on the constant external pressure and the baseline diameters & pressures.
        # Reference diameter does not correspond to the baseline diameter.
        pressure_difference_vertex = flownetwork.pressure - distensibility.pressure_ref
        pressure_difference_edge = .5 * np.sum(pressure_difference_vertex[flownetwork.edge_list], axis=1)[eids_dist]

        # Solve cubic formula for diameter ref (x**3 - b*x**2 + c = 0), b = kappa & c = kappa * Diameter_baseline
        # Using the sympy library, compute the symbolic solution for this equation - 1 real and 2 complex solutions
        from scipy import optimize
        import math
        kappa = distensibility.e_modulus * distensibility.wall_thickness / pressure_difference_edge
        b = kappa; c = kappa * np.square(flownetwork.diameter[eids_dist])
        median = np.median(flownetwork.diameter[eids_dist])
        order = int(np.abs(math.log10(median)))  # the order of magnitude of the median
        x0 = np.round(median, order+2)  # initial guess

        solver = lambda b, c: optimize.newton(_f, x0=x0, fprime=_df, args=(b, c))  # newton method
        vec_solver = np.vectorize(solver)  # vectorize the solver - b, c are arrays
        diameter_ref = vec_solver(b, c)  # get diameter ref

        if True in (diameter_ref < .4 * flownetwork.diameter[eids_dist]):
            sys.exit("Error: Suspicious small reference diameter (compared to baseline diameter) detected. ")

        distensibility.diameter_ref = diameter_ref


def _f(x, b, c):
    """
    Equation of computing  diameter ref base on a non-linear p-A relation proposed by Rammos et al. (1998).
    Diameter ref equation
    """
    return x ** 3 + b * (x ** 2) - c


def _df(x, b, c):
    """
    The first derivative of the function
    """
    return 3 * x ** 2 + 2 * b * x
