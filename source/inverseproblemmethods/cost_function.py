# todo: Class to implement cost function. Main idea:

# todo: store arrays in inverse_model_class
#  target_eid = array with all edge ids with target values
#  target_type = array with all types (velocity, flow rate, ev, flow rate mean)
#  target_range = array with range type (precise value or range)
#  target_min_value =
#  target_max_value

# todo: compute cost function value

# todo: derivatives df/dT and df/dp for all cost terms (dT/dalpha handled from parameter_space class)

# todo: This is a construction site
# from abc import ABC, abstractmethod
# from types import MappingProxyType
#
# class CostFunction(ABC):
#     """
#     Abstract base class for the implementations related to the specific cost function
#     """
#
#     def __init__(self, PARAMETERS: MappingProxyType):
#         """
#         Constructor of CostFunction.
#         :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
#         :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
#         """
#         self._PARAMETERS = PARAMETERS
#
#     @abstractmethod
#     def update_cost_function_value(self):
#         pass
#
#     @abstractmethod
#     def get_partial_f_partial_T(self):
#         pass
#
#     @abstractmethod
#     def get_partial_f_partial_p(self):
#         pass
