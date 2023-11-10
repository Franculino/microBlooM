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
import source.fileio.read_distensibility_parameters as read_distensibility_parameters
import source.distensibilitymodules.distensibility_law_initialise as distensibility_law_initialise
import source.distensibilitymodules.distensibility_law_diam_update as distensibility_law_diam_update
import source.strokemodules.ischaemic_stroke_state as ischaemic_stroke_state
import source.fileio.read_autoregulation_parameters as read_autoregulation_parameters
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

    @abstractmethod
    def setup_distensibility_model(self, PARAMETERS):
        """
        Abstract method to set up the distensibility model
        """

    @abstractmethod
    def setup_ischaemic_stroke_model(self, PARAMETERS):
        """
        Abstract method to set up the ischaemic stroke model
        """

    @abstractmethod
    def setup_autoregulation_model(self, PARAMETERS):
        """
        Abstract method to set up the distensibility model
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
            case 3:
                imp_read = read_network.ReadNetworkIgraph(PARAMETERS)  # Imports a graph from igraph file (pickle file)
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
            case 4:
                imp_write = write_network.WriteNetworkCsv(PARAMETERS)  # Writes the results into two csv files
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
                imp_transmiss = transmissibility.TransmissibilityVitroPries(PARAMETERS)
                imp_velocity = rbc_velocity.RbcVelocityFahraeus(PARAMETERS)
            case 3:  # Takes RBCs into account based on the empirical laws by Pries and Secomb (2005)
                imp_hd = discharge_haematocrit.DischargeHaematocritVitroPries2005(PARAMETERS)
                imp_transmiss = transmissibility.TransmissibilityVitroPries(PARAMETERS)
                imp_velocity = rbc_velocity.RbcVelocityFahraeus(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option for the handling of RBCs (rbc_impact_option)")

        # Initialise the classes handling the solution of the linear system (build system and solver)
        match PARAMETERS["solver_option"]:
            case 1:
                imp_buildsystem = build_system.BuildSystemSparseCoo(PARAMETERS)  # Fast approach to build the system
                imp_solver = pressure_flow_solver.PressureFlowSolverSparseDirect(PARAMETERS)  # Direct solver
            case 2:
                imp_buildsystem = build_system.BuildSystemSparseCoo(PARAMETERS)  # Fast approach to build the system
                imp_solver = pressure_flow_solver.PressureFlowSolverPyAMG(PARAMETERS)  # Iterative solver
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
                imp_adjointparameter = adjoint_method_parameters.AdjointMethodImplementationsAbsBoundaryPressure(PARAMETERS)
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
            case 2:
                imp_adjointsolver = adjoint_method_solver.AdjointMethodSolverPyAMG(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option for the solver of the inverse model (inverse_model_solver)")

        return imp_readtargetvalues, imp_readparameters, imp_adjointparameter, imp_adjointsolver, imp_alphamapping

    def setup_distensibility_model(self, PARAMETERS):
        """
        Set up the distensibility model and returns various implementations of the distensibility model
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        :returns: the implementation objects. Error if invalid option is chosen.
        """

        match PARAMETERS["read_dist_parameters_option"]:
            case 1:  # Do not read anything
                imp_read_dist_parameters = read_distensibility_parameters.ReadDistensibilityParametersNothing(PARAMETERS)
            case 2:  # Read distensibility porameters from csv file
                imp_read_dist_parameters = read_distensibility_parameters.ReadDistensibilityParametersFromFile(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option to read distensibility parameters (read_dist_parameters_option)")

        match PARAMETERS["dist_ref_state_option"]:
            case 1:  # No update of diameters due to vessel distensibility
                imp_dist_ref_state = distensibility_law_initialise.DistensibilityInitialiseNothing(PARAMETERS)
            case 2:  # Passive diameter changes, linearised. p_ext = p_base, d_ref = d_base
                imp_dist_ref_state = distensibility_law_initialise.DistensibilityLawPassiveReferenceBaselinePressure(PARAMETERS)
            case 3:  # Passive diameter changes, linearised. p_ext=0, d_ref computed based on Sherwin et al. (2003).
                imp_dist_ref_state = distensibility_law_initialise.DistensibilityLawPassiveReferenceConstantExternalPressureSherwin(PARAMETERS)
            case 4:  # Passive diameter changes, linearised. p_ext=0, d_ref computed based on Urquiza et al. (2006).
                imp_dist_ref_state = distensibility_law_initialise.DistensibilityLawPassiveReferenceConstantExternalPressureUrquiza(PARAMETERS)
            case 5:  # Passive diameter changes, linearised. p_ext=0, d_ref computed based on Rammos et al. (1998).
                imp_dist_ref_state = distensibility_law_initialise.DistensibilityLawPassiveReferenceConstantExternalPressureRammos(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option to define the reference state (dist_ref_state_option)")

        match PARAMETERS["dist_pres_area_relation_option"]:
            case 1:  # No update of diameters due to vessel distensibility
                imp_dist_pres_area_relation = distensibility_law_diam_update.DistensibilityLawUpdateNothing(PARAMETERS)
            case 2:  # Update of diameters based on a non-linear p-A ralation proposed by Sherwin et al. (2003).
                imp_dist_pres_area_relation = distensibility_law_diam_update.DistensibilityLawUpdatePassiveSherwin(PARAMETERS)
            case 3:  # Update of diameters based on a non-linear p-A ralation proposed by Urquiza et al. (2006).
                imp_dist_pres_area_relation = distensibility_law_diam_update.DistensibilityLawUpdatePassiveUrquiza(PARAMETERS)
            case 4:  # Update of diameters based on a linear p-A ralation proposed by Rammos et al. (1998).
                imp_dist_pres_area_relation = distensibility_law_diam_update.DistensibilityLawUpdatePassiveRammos(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option to define the p-A ralation (dist_pres_area_relation_option)")

        return imp_read_dist_parameters, imp_dist_ref_state, imp_dist_pres_area_relation


    def setup_ischaemic_stroke_model(self, PARAMETERS):
        """
        Set up the ischaemic stroke model and returns various implementations of the stroke model
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        :returns: the implementation objects. Error if invalid option is chosen.
        """
        match PARAMETERS["simulate_ischaemic_stroke_option"]:
            case 1:  # Not induce stroke
                imp_sim_ischaemic_stroke = ischaemic_stroke_state.StrokeStateNothing(PARAMETERS)
            case 2:  # Induce stroke in a hexagonal network
                imp_sim_ischaemic_stroke = ischaemic_stroke_state.StrokeStateHexagonalNetwork(PARAMETERS)
            case 3:  # Induce stroke in a network reading diameters at stroke state from a csv file
                imp_sim_ischaemic_stroke = ischaemic_stroke_state.StrokeStateDiametersCSVFile(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option to simulate stroke (simulate_ischaemic_stroke_option)")

        return imp_sim_ischaemic_stroke


    def setup_autoregulation_model(self, PARAMETERS):
        """
        Set up the autoregulation model and returns various implementations of the autoregulation model
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        :returns: the implementation objects. Error if invalid option is chosen.
        """

        match PARAMETERS["read_auto_parameters_option"]:
            case 1:  # Do not read anything
                imp_read_auto_parameters = read_autoregulation_parameters.ReadAutoregulationParametersNothing(PARAMETERS)
            case 2:  # Read distensibility porameters from csv file
                imp_read_auto_parameters = read_autoregulation_parameters.ReadAutoregulationParametersFromFile(PARAMETERS)
            case _:
                sys.exit("Error: Choose valid option to read autoregulation parameters (read_auto_parameters_option)")

        match PARAMETERS["dist_ref_state_option"]:
            case 1:  # No update of diameters due to vessel distensibility
                imp_dist_ref_state = distensibility_law.DistensibilityInitialiseNothing(PARAMETERS)
            case 2:  # Passive diameter changes, linearised. p_ext = p_base, d_ref = d_base
                imp_dist_ref_state = distensibility_law.DistensibilityLawPassiveReferenceBaselinePressure(PARAMETERS)

        return imp_read_auto_parameters
