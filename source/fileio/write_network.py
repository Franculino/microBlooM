from abc import ABC, abstractmethod
from types import MappingProxyType
import igraph
from copy import deepcopy
import numpy as np
import sys


class WriteNetwork(ABC):
    """
    Abstract base class for the implementations related to writing the results to files.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of WriteNetwork.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def write(self, flownetwork):
        """
        Write the network and results to a file
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class WriteNetworkNothing(WriteNetwork):
    """
    Class for not writing anything
    """

    def write(self, flownetwork):
        """
        Do not write anything
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """


class WriteNetworkIgraph(WriteNetwork):
    """
    Class for writing the results to igraph format.
    """

    def write(self, flownetwork):
        """
        Write the network and simulation data into an igraph file
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        if self._PARAMETERS["write_override_initial_graph"]:
            graph = igraph.Graph(flownetwork.edge_list.tolist())
            print("Warning: Currently a graph is always overwritten...")
            pass
        else:
            edge_list = flownetwork.edge_list
            graph = igraph.Graph(edge_list.tolist())  # Generate igraph based on edge_list

        if flownetwork.diameter is not None:
            graph.es["diameter"] = flownetwork.diameter

        if flownetwork.length is not None:
            graph.es["length"] = flownetwork.length

        if flownetwork.flow_rate is not None:
            graph.es["flow_rate"] = flownetwork.flow_rate

        if flownetwork.rbc_velocity is not None:
            graph.es["rbc_velocity"] = flownetwork.rbc_velocity

        if flownetwork.ht is not None:
            graph.es["ht"] = flownetwork.ht

        graph.vs["xyz"] = flownetwork.xyz.tolist()

        if flownetwork.pressure is not None:
            graph.vs["pressure"] = flownetwork.pressure

        graph.write_pickle(self._PARAMETERS["write_path_igraph"])
        # todo: check that old graph is not overwritten
        # todo: handle boundaries


class WriteNetworkVtp(WriteNetwork):
    """
    Class for writing the results to vtp format. Can be used to visualise results in paraview.
    Function taken from Franca/Chryso. Todo: Documentation
    """

    def _write_array(self, f, array, name, zeros=0, verbose=False):
        """
        Print arrays with different number of components, setting NaNs to 'substitute'.
        Optionally, a given number of zero-entries can be prepended to an
        array. This is required when the graph contains unconnected vertices.
        Function taken from Franca/Chryso
        """
        tab = "  "
        space = 5 * tab
        substituteD = -1000.
        substituteI = -1000
        zeroD = 0.
        zeroI = 0

        array_dimension = np.size(np.shape(array))

        if array_dimension > 1:  # For arrays where attributes are vectors (e.g. coordinates)
            noc = np.shape(array)[1]
            firstel = array[0][0]
            Nai = len(array)
            Naj = np.ones(Nai, dtype=np.int) * noc
        else:
            noc = 1
            firstel = array[0]
            Nai = len(array)
            Naj = np.array([0], dtype='int')

        if type(firstel) == str:
            if verbose:
                print("WARNING: array '%s' contains data of type 'string'!" % name)
            return  # Cannot have string-representations in paraview.

        if "<type 'NoneType'>" in map(str, np.unique(np.array(map(type, array)))):
            if verbose:
                print("WARNING: array '%s' contains data of type 'None'!" % name)
            return

        if any([type(firstel) == x for x in
                [float, np.float32, np.float64, np.longdouble]]):
            atype = "Float64"
            format = "%f"
        elif any([type(firstel) == x for x in
                  [int, np.int8, np.int16, np.int32, np.int64]]):
            atype = "Int64"
            format = "%i"
        else:
            if verbose:
                print("WARNING: array '%s' contains data of unknown type!" % name)
                print("k1")
            return

        f.write('{}<DataArray type="{}" Name="{}" '.format(4 * tab, atype, name))
        f.write('NumberOfComponents="{}" format="ascii">\n'.format(noc))

        if noc == 1:
            if atype == "Float64":
                for i in range(zeros):
                    f.write('{}{}\n'.format(space, zeroD))
                aoD = np.array(array, dtype='double')
                for i in range(Nai):
                    if not np.isfinite(aoD[i]):
                        f.write('{}{}\n'.format(space, substituteD))
                    else:
                        f.write('{}{}\n'.format(space, aoD[i]))
            elif atype == "Int64":
                for i in range(zeros):
                    f.write('{}{}\n'.format(space, zeroI))
                aoI = np.array(array, dtype=np.int64)
                for i in range(Nai):
                    if not np.isfinite(aoI[i]):
                        f.write('{}{}\n'.format(space, substituteI))
                    else:
                        f.write('{}{}\n'.format(space, aoI[i]))
        else:
            if atype == "Float64":
                atD = np.array(array, dtype='double')
                for i in range(zeros):
                    f.write(space)
                    for j in range(Naj[0]):
                        f.write('{} '.format(zeroD))
                    f.write('\n')
                for i in range(Nai):
                    f.write(space)
                    for j in range(Naj[i]):
                        if not np.isfinite(atD[i, j]):
                            f.write('{} '.format(substituteD))
                        else:
                            f.write('{} '.format(atD[i, j]))
                    f.write('\n')
            elif atype == "Int64":
                atI = np.array(array, dtype=np.int32)
                for i in range(zeros):
                    f.write(space)
                    for j in range(Naj[0]):
                        f.write('{} '.format(zeroI))
                    f.write('\n')
                for i in range(Nai):
                    f.write(space)
                    for j in range(Naj[i]):
                        if not np.isfinite(atI[i, j]):
                            f.write('{} '.format(substituteI))
                        else:
                            f.write('{}'.format(atI[i, j]))
                    f.write('\n')
        f.write('{}</DataArray>\n'.format(4 * tab))

    def write(self, flownetwork):
        """
        Write the network and simulation data into a vtp file (e.g. for paraview)
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

        # Convert flownetwork object to igraph file, which can then be exported to vtp

        edge_list = flownetwork.edge_list
        graph = igraph.Graph(edge_list.tolist())  # Generate igraph based on edge_list

        if flownetwork.diameter is not None:
            graph.es["diameter"] = flownetwork.diameter

        if flownetwork.length is not None:
            graph.es["length"] = flownetwork.length

        if flownetwork.flow_rate is not None:
            graph.es["flow_rate"] = flownetwork.flow_rate

        if flownetwork.rbc_velocity is not None:
            graph.es["rbc_velocity"] = flownetwork.rbc_velocity

        if flownetwork.ht is not None:
            graph.es["ht"] = flownetwork.ht

        graph.vs["xyz"] = flownetwork.xyz.tolist()

        if flownetwork.pressure is not None:
            graph.vs["pressure"] = flownetwork.pressure

        # Make a copy of the graph so that modifications are possible, without
        # changing the original. Add indices that can be used for comparison with
        # the original, even after some edges / vertices in the copy have been
        # deleted:
        G = deepcopy(graph)
        G.vs['index'] = range(G.vcount())
        if G.ecount() > 0:
            G.es['index'] = range(G.ecount())

        # Delete selfloops as they cannot be viewed as straight cylinders and their
        # 'angle' property is 'nan':
        G.delete_edges(np.nonzero(G.is_loop())[0].tolist())

        tab = "  "
        fname = self._PARAMETERS["write_path_igraph"]
        f = open(fname, 'w')

        # Find unconnected vertices:
        unconnected = np.nonzero([x == 0 for x in G.strength(weights=
                                                             [1 for i in range(G.ecount())])])[0].tolist()

        # Header
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PolyData" version="0.1" ')
        f.write('byte_order="LittleEndian">\n')
        f.write('{}<PolyData>\n'.format(tab))
        f.write('{}<Piece NumberOfPoints="{}" '.format(2 * tab, G.vcount()))
        f.write('NumberOfVerts="{}" '.format(len(unconnected)))
        f.write('NumberOfLines="{}" '.format(G.ecount()))
        f.write('NumberOfStrips="0" NumberOfPolys="0">\n')

        # Vertex data
        keys = G.vs.attribute_names()
        keysToRemove = []  # Currently no keys are removed. May be changed later.
        # keysToRemove = ['coords', 'pBC', 'rBC', 'kind', 'sBC', 'inflowE', 'outflowE', 'adjacent', 'mLocation',
        #                 'lDir', 'diameter']
        for key in keysToRemove:
            if key in keys:
                keys.remove(key)
        f.write('{}<PointData Scalars="Scalars_p">\n'.format(3 * tab))

        for key in keys:
            self._write_array(f, G.vs[key], key, verbose=True) # verbose = verbose
        f.write('{}</PointData>\n'.format(3 * tab))

        # Edge data
        keys = G.es.attribute_names()
        keysToRemove = [] # Currently no keys are removed. May be changed later.
        # keysToRemove = ['diameters', 'lengths', 'points', 'rRBC', 'tRBC', 'connectivity']
        for key in keysToRemove:
            if key in keys:
                keys.remove(key)
        f.write('{}<CellData Scalars="diameter">\n'.format(3 * tab))

        for key in keys:
            self._write_array(f, G.es[key], key, zeros=len(unconnected), verbose=True) # verbose = verbose
        f.write('{}</CellData>\n'.format(3 * tab))

        # Vertices
        f.write('{}<Points>\n'.format(3 * tab))
        self._write_array(f, np.vstack(G.vs["xyz"]), "xyz", verbose=True) # verbose = verbose
        f.write('{}</Points>\n'.format(3 * tab))

        # Unconnected vertices
        if unconnected != []:
            f.write('{}<Verts>\n'.format(3 * tab))
            f.write('{}<DataArray type="Int64" '.format(4 * tab))
            f.write('Name="connectivity" format="ascii">\n')
            for vertex in unconnected:
                f.write('{}{}\n'.format(5 * tab, vertex))
            f.write('{}</DataArray>\n'.format(4 * tab))
            f.write('{}<DataArray type="Int64" '.format(4 * tab))
            f.write('Name="offsets" format="ascii">\n')
            for i in range(len(unconnected)):
                f.write('{}{}\n'.format(5 * tab, 1 + i))
            f.write('{}</DataArray>\n'.format(4 * tab))
            f.write('{}</Verts>\n'.format(3 * tab))

        # Edges
        f.write('{}<Lines>\n'.format(3 * tab))
        f.write('{}<DataArray type="Int64" '.format(4 * tab))
        f.write('Name="connectivity" format="ascii">\n')
        for edge in G.get_edgelist():
            f.write('{}{} {}\n'.format(5 * tab, edge[0], edge[1]))
        f.write('{}</DataArray>\n'.format(4 * tab))
        f.write('{}<DataArray type="Int64" '.format(4 * tab))
        f.write('Name="offsets" format="ascii">\n')
        for i in range(G.ecount()):
            f.write('{}{}\n'.format(5 * tab, 2 + i * 2))
        f.write('{}</DataArray>\n'.format(4 * tab))
        f.write('{}</Lines>\n'.format(3 * tab))

        # Footer
        f.write('{}</Piece>\n'.format(2 * tab))
        f.write('{}</PolyData>\n'.format(1 * tab))
        f.write('</VTKFile>\n')

        f.close()