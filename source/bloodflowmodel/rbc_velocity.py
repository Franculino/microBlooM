from abc import ABC, abstractmethod
from types import MappingProxyType
import sys
import numpy as np


class RbcVelocity(ABC):
    """
    Abstract base class for the implementations related to calculating the red blood cell velocity
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of RbcVelocity
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS


    @abstractmethod
    def update_velocity(self, flownetwork):
        """
        Abstract method to update the red blood flow velocity
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    def _get_bulk_flow_velocity(self, flownetwork):
        """
        Calculate the bulk flow velocity.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :return: Bulk flow velocity in every edge
        :rtype: 1d numpy array
        """

        diameter = flownetwork.diameter
        flow_rate = flownetwork.flow_rate
        flownetwork.bulk_flow_velocity = flow_rate / (np.square(diameter) * np.pi / 4)
        pos_bulk_velocity = abs(flownetwork.bulk_flow_velocity)
        flownetwork.null_velocity_edges = np.where(pos_bulk_velocity < 1e-6)
        return flow_rate / (np.square(diameter) * np.pi / 4)


class RbcVelocityBulk(RbcVelocity):
    """
    Class for calculating the red blood cell velocity based on the bulk flow velocity. Ignore Fahraeus effect.
    """

    def update_velocity(self, flownetwork):
        """
        Calculate the red blood flow velocity without considering the Fahraeus effect. Red blood cell velocity is
        identical to the bulk flow velocity
        :param flownetwork: flow network object
        :type: source.flow_network.FlowNetwork
        """
        flownetwork.rbc_velocity = self._get_bulk_flow_velocity(flownetwork)


class RbcVelocityFahraeus(RbcVelocity):
    """
    Class for calculating the red blood cell velocity with Fahraeus effect.
    """

    def update_velocity(self, flownetwork):
        """
        Calculate the red blood flow velocity considering the Fahraeus effect. Red blood cell velocity is
        bulk velocity * hd / ht
        :param flownetwork: flow network object
        :type: source.flow_network.FlowNetwork
        """
        hd = flownetwork.hd
        ht = flownetwork.ht

        # Make sure that if ht=0, the ratio hd/ht is 1
        hd_ht_ratio = np.ones(flownetwork.nr_of_es)
        hd_ht_ratio[flownetwork.ht > 0.] = flownetwork.hd[flownetwork.ht > 0.] / flownetwork.ht[flownetwork.ht > 0.]

        flownetwork.rbc_velocity = self._get_bulk_flow_velocity(flownetwork) * hd_ht_ratio

