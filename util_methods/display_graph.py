"""
To display a graph from pkl format
"""
import pickle
from scipy.sparse import *
from scipy.sparse.linalg import spsolve
import numpy as np
import igraph as ig
import matplotlib.pyplot as plt


# from write_vtp_py3 import write_vtp

def unpack_pickle(path):
    with open(path, 'rb') as f:
        unpacked = pickle.load(f)
    return unpacked


def display_graph(g):
    """
    Function to display the graph and give particulary color to each part
    :param g: graph created with graph_creation()
    :return: None
    """

    # create the figure for the plot
    fig, ax = plt.subplots(figsize=(25, 25))

    ig.plot(
        # graph to be shown
        g,
        bbox=[1500, 1500],
        # margin=100,
        target=ax,  # target="graph.pdf",
        # layout="auto",
        # size of the vertex
        #vertex_size=50,
        vertex_size= 0.3,
        # to color the different vertex, also an example to conditionally color them in igraph
        vertex_color=["lightblue"],
        # width of nodes outline
        vertex_frame_width=1.4,
        # color of the vertex outline
        vertex_frame_color="black",
        # nodes label
        vertex_label=[[str(round(g.vs[i]["pressure"], 3)), i] for i in range(0, len(g.vs["pressure"]))],
        # size of nodes label
        vertex_label_size=10.0,
        # size of edges
        edge_width=1,
        # color of edges
        edge_color=["steelblue"],  # if lens == 5e-06 else "red" for lens in g.es["diameter"]],
        # edge labels
        # edge_label=[(str((g.es[i]["ht"]))) for i in  range(0, len(g.es["ht"]))],
        # edge_label=[round(num, 3) for num in g.es["ht"]] ,#str(round(g.es["ht"], 3)),  round(num, 1) for num in g.es["ht"]
        # edge_label= [i for i in range(0, len(g.vs))],
        edge_label=[str("Q:" + "{:.2e}".format(g.es[i]["flow_rate"])) + "\n \n RBCs:" + str(
            "{:.2e}".format(np.abs(g.es["flow_rate"][i]) * g.es["hd"][i])) + "\n \n" + str(
            "HD:" + "{:.2e}".format(g.es[i]["hd"])) + "\n \n " + str(i) for i in range(0, len(g.es["flow_rate"]))],
        # edge label size
        edge_label_size=10.0,
        edge_align_label=False,
    )
    plt.show()


def display_graph_util():
    g = unpack_pickle("/Users/cucciolo/Desktop/microBlooM/data/out/hematocrit.pkl")
    #    g.to_directed("acyclic")
    print(g.is_directed())
    display_graph(g)


if __name__ == "__main__":
    display_graph_util()
