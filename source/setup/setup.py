from abc import ABC, abstractmethod
from types import MappingProxyType
import source.fileio.read_network as read_network
import source.fileio.write_network as write_network
import source.bloodflowmodel.tube_haematocrit as tube_haematocrit
import source.bloodflowmodel.discharge_haematocrit as discharge_haematocrit
import source.bloodflowmodel.transmissibility as transmissibility
import source.bloodflowmodel.pressure_flow_solver as pressure_flow_solver
import source.bloodflowmodel.build_system as build_system
import source.bloodflowmodel.rbc_velocity as rbc_velocity
import source.fileio.read_target_values as read_target_values
import source.fileio.read_parameters as read_parameters
import source.inverseproblemmodules.adjoint_method_implementations as adjoint_method_parameters
import source.inverseproblemmodules.adjoint_method_solver as adjoint_method_solver
import source.inverseproblemmodules.alpha_restriction as alpha_mapping
import sys


class Setup(ABC):
    """
    Abstract base class for the setup of the simulation
    """

    @abstractmethod
    def setup_bloodflow_model(self, PARAMETERS):
        """
        Abstract method to set up the simulation
        """

    @abstractmethod
    def setup_inverse_model(self, PARAMETERS):
        """
        Abstract method to set up the inverse model
        """


class SetupSimulation(Setup):
    """
    Class for setting up a simulation that only includes the blood flow model
    """
    def setup_bloodflow_model(self, PARAMETERS):
        """
        Set up the simulation and returns various implementations of the blood flow model
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        :returns: the implementation objects. Error if invalid option is chosen. todo return specification
        """

        # Initialise the class to read / generate a network
        match PARAMETERS["read_network_option"]:
            case 1:
                imp_read = read_network.ReadNetworkHexagonal(PARAMETERS)  # Initialises a hexagonal 2D network
            case 2:
                imp_read = read_network.ReadNetworkCsv(PARAMETERS)  # Imports an arbitrary network from csv files
            case _:
                sys.exit("Error: Choose valid option to generate or import a network (read_network_option)")

        # Initialise the class to write the network to a file (or not)
        match PARAMETERS["write_network_option"]:
            case 1:
                imp_write = write_network.WriteNetworkNothing(PARAMETERS)  # Does not do anything
            case 2:
                imp_write = write_network.WriteNetworkIgraph(PARAMETERS)  # Writes the results into an igraph pkl file
            case 3:
                imp_write = write_network.WriteNetworkVtp(PARAMETERS)  # Writes the results into an igraph pkl file
            case _:
                sys.exit("Error: Choose valid option to write a network to file (write_network_option)")

        # Initialise the class to specify how the tube haematocrit is computed / specified
        match PARAMETERS["tube_haematocrit_option"]:
            case 1:
                imp_ht = tube_haematocrit.TubeHaematocritNewtonian(PARAMETERS)  # Neglects the impact of RBCs (ht = 0)
            case 2:
                imp_ht = tube_haematocrit.TubeHaematocritConstant(PARAMETERS)  # Constant ht for all edges
            case _:
                sys.exit("Error: Choose valid option for the tube haematocrit (tube_haematocrit_option)")

        # Initialise the classes handling the impact of RBCs on discharge haematocrit, transmissibility and RBC velocity
        match PARAMETERS["rbc_impact_option"]:
            case 1:  # Does not take the impact of RBCs into account
                imp_hd = discharge_haematocrit.DischargeHaematocritNewtonian(PARAMETERS)  # Assumes hd = 0
                imp_transmiss = transmissibility.TransmissibilityPoiseuille(PARAMETERS)  # Assumes mu_rel = 0
                imp_velocity = rbc_velocity.RbcVelocityBulk(PARAMETERS)  # No Fahraeus effect (u_RBC = u_Bulk)
            case 2:  # Takes RBCs into account based on the empirical laws by Pries, Neuhaus, Gaehtgens (1992)
                imp_hd = discharge_haematocrit.DischargeHaematocritVitroPries1992(PARAMETERS)
                imp_transmiss = transmissibility.TransmissibilityVitroPries1992(PARAMETERS)
                imp_velocity = rbc_velocity.RbcVelocityFahraeus(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option for the handling of RBCs (rbc_impact_option)")

        # Initialise the classes handling the solution of the linear system (build system and solver)
        match PARAMETERS["solver_option"]:
            case 1:
                imp_buildsystem = build_system.BuildSystemSparseCoo(PARAMETERS)  # Fast approach to build the system
                imp_solver = pressure_flow_solver.PressureFlowSolverSparseDirect(PARAMETERS)  # Direct solver
            case _:
                sys.exit("Error: Choose valid option for the solver (solver_option)")

        return imp_read, imp_write, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, imp_solver


    def setup_inverse_model(self, PARAMETERS):
        """
        Set up the inverse model and returns various implementations of the inverse model
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        :returns: the implementation objects. Error if invalid option is chosen.
        """

        match PARAMETERS["parameter_space"]:
            case 1:  # Tune relative diameters
                imp_adjointparameter = adjoint_method_parameters.AdjointMethodImplementationsRelDiam(PARAMETERS)
                imp_readtargetvalues = read_target_values.ReadTargetValuesEdge(PARAMETERS)
                imp_readparameters = read_parameters.ReadParametersEdges(PARAMETERS)
            case 2:  # Tune relative transmissibilities
                imp_adjointparameter = adjoint_method_parameters.AdjointMethodImplementationsRelTransmiss(PARAMETERS)
                imp_readtargetvalues = read_target_values.ReadTargetValuesEdge(PARAMETERS)
                imp_readparameters = read_parameters.ReadParametersEdges(PARAMETERS)
            case 11:  # Tune boundary pressures
                imp_adjointparameter = adjoint_method_parameters.AdjointMethodImplementationsAbsBoundaryPressure(
                    PARAMETERS)
                imp_readtargetvalues = read_target_values.ReadTargetValuesEdge(PARAMETERS)
                imp_readparameters = read_parameters.ReadParametersVertices(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option for the parameter space (parameter_space)")

        match PARAMETERS["parameter_restriction"]:
            case 1:
                imp_alphamapping = alpha_mapping.AlphaRestrictionLinear(PARAMETERS)
            case 2:
                imp_alphamapping = alpha_mapping.AlphaMappingTanh(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option for the restriction of the parameter (parameter_restriction)")

        match PARAMETERS["inverse_model_solver"]:
            case 1:
                imp_adjointsolver = adjoint_method_solver.AdjointMethodSolverSparseDirect(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option for the solver of the inverse model (inverse_model_solver)")

        return imp_readtargetvalues, imp_readparameters, imp_adjointparameter, imp_adjointsolver, imp_alphamapping
