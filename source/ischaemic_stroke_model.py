from types import MappingProxyType

import source.strokemodules.ischaemic_stroke_state as stroke_state

import source.flow_network as flownetwork
import source.distensibility as distensibility


class IschaemicStrokeModel(object):

    def __init__(self, flownetwork: flownetwork.FlowNetwork,
                 distensibility: distensibility.Distensibility,
                 imp_sim_ischaemic_stroke: stroke_state.IschaemicStrokeState,
                 PARAMETERS: MappingProxyType):

        # "Reference" to flow network and distensibility models
        self._flownetwork = flownetwork
        self._distensibility = distensibility

        # "References" to implementations
        self._imp_sim_ischaemic_stroke = imp_sim_ischaemic_stroke

        # "Reference" to parameter dict
        self._PARAMETERS = PARAMETERS

        # Modelling parameters
        self.diameters_stroke = None  # diameters at stroke state for the entire network

        return

    def simulate_ischaemic_stroke(self):
        """
        Simulate ischaemic stroke by changing graph attributes (e.g., diameters) at stroke state.
        """
        self._imp_sim_ischaemic_stroke.induce_ischaemic_stroke(self, self._flownetwork, self._distensibility)