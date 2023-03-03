import sys
from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import igraph


class ReadNetwork(ABC):
    """
    Abstract base class for the implementations related to generating or importing a vascular network.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of ReadNetwork.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def read(self, flownetwork):
        """
        Update flownetwork based on a newly generated or imported vascular network.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class ReadNetworkHexagonal(ReadNetwork):
    """
    Class for generating a hexagonal network.
    """

    def read(self, flownetwork):
        """
        Generate a hexagonal network with constant vessel diameters and lengths. Update the flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        vessel_diameter = self._PARAMETERS["hexa_diameter"]
        vessel_length = self._PARAMETERS["hexa_edge_length"]
        nr_of_hexagon_x = np.int(self._PARAMETERS["nr_of_hexagon_x"])
        nr_of_hexagon_y = np.int(self._PARAMETERS["nr_of_hexagon_y"])

        if nr_of_hexagon_x % 2 == 0 or nr_of_hexagon_y % 2 == 0:
            raise SystemExit("Error: Choose odd numer of hexagon.")

        nr_vs_x = nr_of_hexagon_x + 1  # Number of vertices in x direction
        nr_vs_y = nr_of_hexagon_y * 2 + 1  # Number of vertices in y direction
        xyz_vs = np.zeros((nr_vs_x * nr_vs_y, 3), dtype=np.float)  # Coordinates x, y, z of vertices

        # Create vertices, calculate xyz coordinates:
        for ii in range(nr_vs_x * nr_vs_y):  # Loop over all vertex indices. Current index ii.
            i = ii % nr_vs_x  # Current vertex index in x-direction (ii = j*n_vs_x + i).
            j = ii // nr_vs_x  # Current vertex index in y-direction.

            xyz_vs[ii, 1] = j * vessel_length / 2 * np.sqrt(3)  # y coordinate of current vertex.
            # Compute x coordinate of vertex.
            if j % 2 == 0:  # For even j.
                if i == 0:
                    xyz_vs[ii, 0] = vessel_length / 2
                elif i % 2 == 1:
                    xyz_vs[ii, 0] = xyz_vs[ii - 1, 0] + vessel_length
                else:
                    xyz_vs[ii, 0] = xyz_vs[ii - 1, 0] + 2 * vessel_length
            else:  # For odd j.
                if i == 0:
                    xyz_vs[ii, 0] = 0
                elif i % 2 == 1:
                    xyz_vs[ii, 0] = xyz_vs[ii - 1, 0] + 2 * vessel_length
                else:
                    xyz_vs[ii, 0] = xyz_vs[ii - 1, 0] + vessel_length

        # Generate edges; connect vertices by edge list
        nr_of_edges = nr_vs_x // 2 * (3 * (nr_vs_y - 1) + 1) - (nr_vs_y - 1) // 2
        edge_list = np.ones((nr_of_edges, 2), dtype=np.int) * (-1)  # initialise edge list with -1
        eid = 0
        # Generate the edges that horizontally connect vertices.
        for ii in range(nr_vs_x * nr_vs_y):  # loop over all vertex indices
            i = ii % nr_vs_x  # Current vertex index in x-direction (ii = j*n_vs_x + i).
            j = ii // nr_vs_x  # Current vertex index in y-direction.

            # For all existing horizontal edges, identify the corresponding vertex indices.
            if (j % 2 == 0 and i % 2 == 0 and i < nr_vs_x-1) or (j % 2 == 1 and i % 2 == 1 and i < nr_vs_x-2):
                edge_list[eid, 0] = ii
                edge_list[eid, 1] = ii + 1
                eid += 1

        # Generate the edges that diagonally connect vertices.
        for ii in range(nr_vs_x * nr_vs_y):  # Loop over all vertex indices
            i = ii % nr_vs_x  # Current vertex index in x-direction (ii = j*n_vs_x + i).
            j = ii // nr_vs_x  # Current vertex index in y-direction.

            # For all existing diagonal edges, identify the corresponding vertex indices.
            if (i % 2 == 1 and j % 2 == 0) or (i % 2 == 0 and j % 2 == 1):
                ii_tr = ii + nr_vs_x  # Index of top-right vertex of ii
                ii_br = ii - nr_vs_x  # Index of bottom-right vertex of ii

                if ii_tr < nr_vs_x * nr_vs_y:  # Only connect to existing vertices (exclude non-existent vertices)
                    edge_list[eid, 0] = ii
                    edge_list[eid, 1] = ii_tr
                    eid += 1
                if ii_br > -1:  # Only connect to existing vertices (exclude non-existent vertices)
                    edge_list[eid, 0] = ii
                    edge_list[eid, 1] = ii_br
                    eid += 1

        # Sort edge_list such that always lower index is in first column.
        edge_list = np.sort(edge_list, axis=1)

        # Sort edge_list based on first column.
        edge_list = edge_list[edge_list[:, 0].argsort()]

        # Assign data to flownetwork class
        # Network attributes
        flownetwork.nr_of_vs = nr_vs_x * nr_vs_y
        flownetwork.nr_of_es = nr_of_edges

        # Edge attributes
        flownetwork.length = np.ones(nr_of_edges) * vessel_length
        flownetwork.diameter = np.ones(nr_of_edges) * vessel_diameter
        flownetwork.edge_list = edge_list

        # Vertex attributes
        flownetwork.xyz = xyz_vs

        # Boundaries, sort according to ascending vertex ids
        import pandas as pd
        df_boundaries = pd.DataFrame({'vs_ids': np.array(self._PARAMETERS["hexa_boundary_vertices"], dtype=np.int),
                        'vals': np.array(self._PARAMETERS["hexa_boundary_values"], dtype=np.float),
                        'types': np.array(self._PARAMETERS["hexa_boundary_types"], dtype=np.int)})

        df_boundaries = df_boundaries.sort_values('vs_ids')

        flownetwork.boundary_vs = df_boundaries["vs_ids"].to_numpy()
        flownetwork.boundary_val = df_boundaries["vals"].to_numpy()
        flownetwork.boundary_type = df_boundaries["types"].to_numpy()


class ReadNetworkCsv(ReadNetwork):
    """
    Class for reading a vascular network from a csv file
    """

    def read(self, flownetwork):
        """
        Import a network from the three csv text files containing vertex, edge and boundary data.

        Vertex (vx) data: At least three columns are required to describe the x, y and z coordinates of all vertices. A
        header for each column has to be provided. Example file structure (order of columns does not matter; additional
        columns are ignored):

        x_coord,y_coord,z_coord
        x_coord_of_vx_0,y_coord_of_vx_0,z_coord_of_vx_0
        x_coord_of_vx_1,y_coord_of_vx_1,z_coord_of_vx_1
                :      ,        :      ,        :
                :      ,        :      ,        :

        Edge data: At least four columns are required to describe the incidence vertices (requires two columns, i.e.,
        one for each incidence vertex per edge), diameters and lengths of all edges. A header for each column has to be
        provided. Example file structure (order of columns does not matter; additional columns are ignored):

        vx_incident_1,vertex_incident_2,diameter,length
        incident_vx_1_of_edge_0,incident_vx_2_of_edge_0,diameter_of_edge_0,length_of_edge_0
        incident_vx_1_of_edge_1,incident_vx_2_of_edge_1,diameter_of_edge_1,length_of_edge_1
                    :          ,            :          ,            :     ,         :
                    :          ,            :          ,            :     ,         :

        Boundary data: At least three columns are required to prescribe the vertex indices of boundary conditions,
        the boundary type (1: pressure, 2: flow rate) and the boundary values (can be pressure or flow rate).
        Example file structure (order of columns does not matter; additional columns are ignored):

        vx_id_of_boundary,boundary_type,boundary_value
        vx_id_boundary_0,boundary_type_0,boundary_value_0
        vx_id_boundary_1,boundary_type_1,boundary_value_1
                :       ,       :       ,       :
                :       ,       :       ,       :

        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        import pandas as pd
        # Extract file path of network files
        path_edge_data = self._PARAMETERS["csv_path_edge_data"]
        path_vertex_data = self._PARAMETERS["csv_path_vertex_data"]
        path_boundary_data = self._PARAMETERS["csv_path_boundary_data"]

        # Read files with pandas
        df_edge_data = pd.read_csv(path_edge_data)
        df_vertex_data = pd.read_csv(path_vertex_data)
        df_boundary_data = pd.read_csv(path_boundary_data)

        # Assign data to flownetwork class
        # Edge attributes
        flownetwork.diameter = df_edge_data[self._PARAMETERS["csv_diameter"]].to_numpy()
        flownetwork.length = df_edge_data[self._PARAMETERS["csv_length"]].to_numpy()
        # Create edge list from the two columns containing the incident vertices of each edge
        edge_list = np.vstack([df_edge_data[self._PARAMETERS["csv_edgelist_v1"]].to_numpy().astype(np.int),
                               df_edge_data[self._PARAMETERS["csv_edgelist_v2"]].to_numpy().astype(np.int)]).transpose()
        flownetwork.edge_list = edge_list

        # Vertex attributes (numpy array containing all dimensions)
        xyz = np.vstack([df_vertex_data[self._PARAMETERS["csv_coord_x"]].to_numpy(),
                         df_vertex_data[self._PARAMETERS["csv_coord_y"]].to_numpy(),
                         df_vertex_data[self._PARAMETERS["csv_coord_z"]].to_numpy()]).transpose()
        flownetwork.xyz = xyz

        # Network attributes
        flownetwork.nr_of_vs = np.size(xyz, 0)
        flownetwork.nr_of_es = np.size(edge_list, 0)

        # Boundaries
        # Sort according to ascending vertex indices of boundaries
        df_boundary_data.sort_values(self._PARAMETERS["csv_boundary_vs"])
        flownetwork.boundary_vs = df_boundary_data[self._PARAMETERS["csv_boundary_vs"]].to_numpy().astype(np.int)
        flownetwork.boundary_type = df_boundary_data[self._PARAMETERS["csv_boundary_type"]].to_numpy().astype(np.int)
        flownetwork.boundary_val = df_boundary_data[self._PARAMETERS["csv_boundary_value"]].to_numpy()


class ReadNetworkIgraph(ReadNetwork):
    def read(self, flownetwork):
        """
        Import a network from igraph file (pickle file)
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Extract file path of network file
        path_igraph = self._PARAMETERS["pkl_path_igraph"]

        # Import pickle
        graph = igraph.Graph.Read_Pickle(path_igraph)

        # Check graph attributes
        current_graph_attributes = graph.attributes()
        current_vertex_attributes = graph.vs.attributes()
        current_edge_attributes = graph.es.attributes()

        target_attributes = ["Mouse", "network_gen_date"]

        target_vert_attri = ["ACA_in", "MCA_in", "COW_in", "coords", "is_AV_root", "is_DA_startingPt",
                             "is_DA_startingPt_added_manually", "is_connected_2caps"]

        target_edge_attri = ["added_manually", "diam_post_exp", "diam_pre_exp", "diameter", "index_exp",
                             "is_collateral", "is_stroke", "length", "type",
                             "vRBC_post_exp", "vRBC_pre_exp", "vRBC_pre_larger10"]

        # deleted:  "diam_post0_exp", "diam_post30_exp", "diam_post60_exp", "diam_post90_exp", "diam_post120_exp",
        #           "vRBC_post0_exp", "vRBC_post0_larger10", "vRBC_post30_exp", "vRBC_post30_larger10", "vRBC_post60_exp",
        #           "vRBC_post60_larger10", "vRBC_post90_exp",     "vRBC_post90_larger10", "vRBC_post120_exp",
        #           "vRBC_post120_larger10",

        attributes_missing_in_graph = []
        attributes_excessive_in_graph = []
        is_consistent = True

        # Check for duplicates in the target_attributes, target_vert_attri & target_edge_attri
        # Sets cannot have two items with the same value. Duplicates Not Allowed.
        if len(set(target_attributes)) != len(target_attributes) or len(set(target_vert_attri)) != len(
                target_vert_attri) or len(set(target_edge_attri)) != len(target_edge_attri):
            is_consistent = False
            print("List of target edge attributes not unique.")
            return is_consistent

        # check if all target attributes are in current graph
        # Graph attributes
        for attr in target_attributes:
            if attr in current_graph_attributes:
                continue
            else:
                attributes_missing_in_graph.append(attr)
                is_consistent = False

        # Vertex attributes
        for attr in target_vert_attri:
            if attr in current_vertex_attributes:
                continue
            else:
                attributes_missing_in_graph.append(attr)
                is_consistent = False

        # Edge attributes
        for attr in target_edge_attri:
            if attr in current_edge_attributes:
                continue
            else:
                attributes_missing_in_graph.append(attr)
                is_consistent = False

        # Check if no additional attributes are in current graph
        # Graph attributes
        for attr in current_graph_attributes:
            if attr in target_attributes:
                continue
            else:
                attributes_excessive_in_graph.append(attr)
                is_consistent = False

        # Vertex attributes
        for attr in current_vertex_attributes:
            if attr in target_vert_attri:
                continue
            else:
                attributes_excessive_in_graph.append(attr)
                is_consistent = False

        # Edge attributes
        for attr in current_edge_attributes:
            if attr in target_edge_attri:
                continue
            else:
                attributes_excessive_in_graph.append(attr)
                is_consistent = False

        if not is_consistent:
            print("Something is wrong with graph attributes...")
            print("The following attributes are missing in current graph:", attributes_missing_in_graph)
            print("The following attributes are excessive in current graph:", attributes_excessive_in_graph)
            import sys
            sys.exit("Warning Message: Check graph attributes")

        print(graph.summary())

        # Assign data to flownetwork class
        # Edge attributes
        flownetwork.diameter = np.array(graph.es["diameter"])
        flownetwork.length = np.array(graph.es["length"])
        flownetwork.edge_list = np.array(graph.get_edgelist())

        # Vertex attributes (numpy array containing all dimensions)
        flownetwork.xyz = np.array(graph.vs["coords"])

        # Network attributes
        flownetwork.nr_of_vs = np.array(graph.vcount())
        flownetwork.nr_of_es = np.array(graph.ecount())

        # Boundaries
        boundary_types = self._PARAMETERS["boundaryType"]
        boundary_values = self._PARAMETERS["boundaryValue"]

        flownetwork.boundary_vs = np.concatenate((np.array(graph.vs(COW_in_eq=1).indices),
                                                  np.array(graph.vs(is_AV_root_eq=1).indices)))

        flownetwork.boundary_type = np.ones(np.size(flownetwork.boundary_vs), dtype=np.int) * boundary_types[1]
        flownetwork.boundary_type[0] = boundary_types[0]

        if boundary_types == [1, 1]:
            # pressure for all AV_roots - outlet
            flownetwork.boundary_val = np.ones(np.size(flownetwork.boundary_vs)) * boundary_values["outlet_pressure"]
            # pressure for COW_in - inlet
            flownetwork.boundary_val[0] = boundary_values["inlet_pressure"]
        elif boundary_types == [2, 1]:
            # pressure for all AV_roots - outlet
            flownetwork.boundary_val = np.ones(np.size(flownetwork.boundary_vs)) * boundary_values["outlet_pressure"]
            # flow rate for COW_in - inlet
            flownetwork.boundary_val[0] = boundary_values["inlet_flow_rate"]
        elif boundary_types == [1, 2]:
            # flow rate for all AV_roots - outlet
            flownetwork.boundary_val = np.ones(np.size(flownetwork.boundary_vs)) * boundary_values["outlet_flow_rate"]
            # pressure for COW_in - inlet
            flownetwork.boundary_val[0] = boundary_values["inlet_pressure"]
        else:
            import sys
            sys.exit("Warning Message - Boundary Conditions: Only flow rate boundary conditions were assigned! "
                     "Define new boundary conditions, including at least one pressure boundary condition!")


class ReadNetworkPlk(ReadNetwork):
    def read(self, flownetwork):
        # todo: import graph from edge_data and vertex_data pickle files.
        # Need example format from franca. that always consistent
        # with open(path_es_dict, "rb") as f:
        #     data_edge = pickle.load(f, encoding="latin1")
        # with open(path_vs_dict, "rb") as f:
        #     data_vertex = pickle.load(f, encoding="latin1")
        pass
