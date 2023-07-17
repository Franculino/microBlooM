import sys
from abc import ABC, abstractmethod
from types import MappingProxyType
import numpy as np
import igraph
import pandas as pd


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
        nr_of_edges = (nr_vs_x // 2 * (3 * (nr_vs_y - 1) + 1) - (nr_vs_y - 1) // 2)
        edge_list = np.ones((nr_of_edges, 2), dtype=np.int) * (-1)  # initialise edge list with -1
        eid = 0
        # Generate the edges that horizontally connect vertices.
        for ii in range(nr_vs_x * nr_vs_y):  # loop over all vertex indices
            i = ii % nr_vs_x  # Current vertex index in x-direction (ii = j*n_vs_x + i).
            j = ii // nr_vs_x  # Current vertex index in y-direction.

            # For all existing horizontal edges, identify the corresponding vertex indices.
            if (j % 2 == 0 and i % 2 == 0 and i < nr_vs_x - 1) or (j % 2 == 1 and i % 2 == 1 and i < nr_vs_x - 2):
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

        edge_list = np.append(edge_list, [[5, 14]], axis=0)
        edge_list = np.append(edge_list, [[13, 22]], axis=0)
        nr_of_edges += 2

        # Sort edge_list such that always lower index is in first column.
        edge_list = np.sort(edge_list, axis=1)

        # Sort edge_list based on first column.
        edge_list = edge_list[edge_list[:, 0].argsort()]

        # Assign data to flownetwork class
        # Network attributes
        flownetwork.nr_of_vs = (nr_vs_x * nr_vs_y)
        flownetwork.nr_of_es = nr_of_edges

        # Edge attributes
        flownetwork.length = np.ones(nr_of_edges) * vessel_length
        flownetwork.diameter = np.ones(nr_of_edges) * vessel_diameter
        flownetwork.edge_list = edge_list

        # Vertex attributes
        flownetwork.xyz = xyz_vs

        # Boundaries, sort according to ascending vertex ids
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
        flownetwork.boundary_val = df_boundary_data[self._PARAMETERS["csv_boundary_value"]].multiply(133.322).to_numpy()  # 133.3223684

        print("Number of  vs: " + str(flownetwork.nr_of_vs))
        print("Number of boundary vs: " + str(len(flownetwork.boundary_vs)))
        print("Number of  es: " + str(flownetwork.nr_of_es))


class ReadNetworkIgraph(ReadNetwork):
    def read(self, flownetwork):
        """
        Import a network from igraph file (pickle file)

        Vertex data: At least one attribute is required to describe the x, y and z coordinates of all vertices.

            one (3 x nv) array, where nv is the number of vertices:
                    [[x0, y0, z0]
                     [x1, y1, z1]
                                :
                                :
                     [xnv, ynv, znv]]

        Edge data: At least two attribute are required to describe the diameters and lengths of all edges.

            two (1 x ne) arrays, where ne is number of edges:
                    diameter: [d0, d1, ..., dne ]
                    length: [l0, l1, ..., lne ]

        Boundary data: At least two vertex attributes are required to prescribe the boundary type
        (1: pressure, 2: flow rate, None: otherwise) and the boundary values (can be pressure or flow rate,
        None: otherwise).

            two (1 x nv) arrays, where nv is number of vertices:
                    boundary_type: [boundary_type_0, boundary_type_1, ..., boundary_type_nv]
                    boundary_value: [boundary_value_0, boundary_value_1, ..., boundary_value_nv]

        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Extract file path of network file
        path_igraph = self._PARAMETERS["pkl_path_igraph"]

        # Import pickle
        graph = igraph.Graph.Read_Pickle(path_igraph)

        print(graph.summary())

        # Assign data to flownetwork class
        # Edge attributes
        flownetwork.diameter = np.array(graph.es[self._PARAMETERS["ig_diameter"]])
        flownetwork.length = np.array(graph.es[self._PARAMETERS["ig_length"]])
        flownetwork.edge_list = np.array(graph.get_edgelist())

        # Vertex attributes (numpy array containing all dimensions)
        flownetwork.xyz = np.array(graph.vs[self._PARAMETERS["ig_coord_xyz"]])

        # Network attributes
        flownetwork.nr_of_vs = graph.vcount()
        flownetwork.nr_of_es = graph.ecount()

        # Boundaries
        boundary_types = np.array(graph.vs[self._PARAMETERS["ig_boundary_type"]])
        flownetwork.boundary_type = boundary_types[np.logical_or(boundary_types == 1, boundary_types == 2)]
        boundary_values = np.array(graph.vs[self._PARAMETERS["ig_boundary_value"]])
        flownetwork.boundary_val = boundary_values[np.logical_or(boundary_types == 1, boundary_types == 2)]
        flownetwork.boundary_vs = np.arange(flownetwork.nr_of_vs)[
            np.logical_or(boundary_types == 1, boundary_types == 2)]


class ReadNetworkPkl(ReadNetwork):
    def read(self, flownetwork):
        # todo: import graph from edge_data and vertex_data pickle files.
        # Need example format from franca. that always consistent
        # with open(path_es_dict, "rb") as f:
        #     data_edge = pickle.load(f, encoding="latin1")
        # with open(path_vs_dict, "rb") as f:
        #     data_vertex = pickle.load(f, encoding="latin1")
        pass


class ReadNetworkSingleHexagon(ReadNetwork):
    """
    Generate a hexagonal network consisting of a single hexagonal with constant vessel diameters and lengths.
    The network has one inflow and two outflows.
    """

    def read(self, flownetwork):
        vessel_diameter = self._PARAMETERS["hexa_diameter"]
        vessel_length = self._PARAMETERS["hexa_edge_length"]

        flownetwork.nr_of_vs = 6

        # EDGE LIST
        edge_list = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
        # Sort edge_list such that always lower index is in first column.
        edge_list = np.sort(edge_list, axis=1)
        # Sort edge_list based on first column.
        edge_list = edge_list[edge_list[:, 0].argsort()]
        flownetwork.nr_of_es = len(edge_list)
        # Edge attributes
        flownetwork.length = np.ones(flownetwork.nr_of_es) * vessel_length
        flownetwork.diameter = np.ones(flownetwork.nr_of_es) * vessel_diameter
        flownetwork.edge_list = edge_list

        df_boundaries = pd.DataFrame(
            {'vs_ids': np.array(self._PARAMETERS["hexa_boundary_vertices"], dtype=np.int),
             'vals': np.array(self._PARAMETERS["hexa_boundary_values"], dtype=np.float),
             'types': np.array(self._PARAMETERS["hexa_boundary_types"], dtype=np.int)})

        df_boundaries = df_boundaries.sort_values('vs_ids')

        flownetwork.boundary_vs = df_boundaries["vs_ids"].to_numpy()
        flownetwork.boundary_val = df_boundaries["vals"].to_numpy()
        flownetwork.boundary_type = df_boundaries["types"].to_numpy()


class ReadNetworkSingleHexagonTrifurcation(ReadNetwork):

    def read(self, flownetwork):
        vessel_diameter = self._PARAMETERS["hexa_diameter"]
        vessel_length = self._PARAMETERS["hexa_edge_length"]

        # EDGE LIST
        # edge_list = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)])
        edge_list = np.array([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 2), (3, 5)])
        # Sort edge_list such that always lower index is in first column.
        edge_list = np.sort(edge_list, axis=1)
        flownetwork.nr_of_vs = 6
        # Sort edge_list based on first column.
        edge_list = edge_list[edge_list[:, 0].argsort()]
        flownetwork.nr_of_es = len(edge_list)
        # Edge attributes
        flownetwork.length = np.ones(flownetwork.nr_of_es) * vessel_length
        flownetwork.diameter = np.ones(flownetwork.nr_of_es) * vessel_diameter
        flownetwork.edge_list = edge_list

        df_boundaries = pd.DataFrame(
            {'vs_ids': np.array(self._PARAMETERS["hexa_boundary_vertices"], dtype=np.int),
             'vals': np.array(self._PARAMETERS["hexa_boundary_values"], dtype=np.float),
             'types': np.array(self._PARAMETERS["hexa_boundary_types"], dtype=np.int)})

        df_boundaries = df_boundaries.sort_values('vs_ids')

        flownetwork.boundary_vs = df_boundaries["vs_ids"].to_numpy()
        flownetwork.boundary_val = df_boundaries["vals"].to_numpy()
        flownetwork.boundary_type = df_boundaries["types"].to_numpy()


class ReadNetworkSingleHexagonDoubleEdge(ReadNetwork):

    def read(self, flownetwork):
        vessel_diameter = self._PARAMETERS["hexa_diameter"]
        vessel_length = self._PARAMETERS["hexa_edge_length"]

        # EDGE LIST
        edge_list = np.array([(0, 1), (1, 2), (1, 2), (2, 3)])
        # Sort edge_list such that always lower index is in first column.
        edge_list = np.sort(edge_list, axis=1)
        flownetwork.nr_of_vs = 4
        # Sort edge_list based on first column.
        edge_list = edge_list[edge_list[:, 0].argsort()]
        flownetwork.nr_of_es = len(edge_list)
        # Edge attributes
        flownetwork.length = np.ones(flownetwork.nr_of_es) * vessel_length
        flownetwork.diameter = np.ones(flownetwork.nr_of_es) * vessel_diameter
        flownetwork.edge_list = edge_list

        df_boundaries = pd.DataFrame(
            {'vs_ids': np.array(self._PARAMETERS["hexa_boundary_vertices"], dtype=np.int),
             'vals': np.array(self._PARAMETERS["hexa_boundary_values"], dtype=np.float),
             'types': np.array(self._PARAMETERS["hexa_boundary_types"], dtype=np.int)})

        df_boundaries = df_boundaries.sort_values('vs_ids')

        flownetwork.boundary_vs = df_boundaries["vs_ids"].to_numpy()
        flownetwork.boundary_val = df_boundaries["vals"].to_numpy()
        flownetwork.boundary_type = df_boundaries["types"].to_numpy()
