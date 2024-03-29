# microBlooM

**MicroBloomM** 
is an open-source simulation framework to study flow characteristics in the microvasculature. The numerical model computes blood flow in microvascular networks for pure plasma flow or considering the impact of red blood cells (Fåhraeus-Linqvist effect) [5,6]. Equations are derived based on Poiseuille’s law and the continuity equation. The microvascular network is represented by a 1D-graph. The elasticity of the blood vessels has been included, allowing the simulation of passive vascular diameter adaptations with respect to pressure changes [7].

Furthermore, different inverse models are available. One version can infer vascular parameters such as vascular diameter and transmissibility based on prescribed flow characteristics [3,4]. Another version can be used to predict pressure boundary conditions required to obtain the prescribed flow characteristics.

The simulations are associated with test cases that can be modified by the user (see [Usage](#usage)). The following list reports the designed test case:

- [`testcase_blood_flow_model.py`](https://github.com/Franculino/microBlooM/blob/main/testcases/testcase_blood_flow_model.py): stationary blood flow in microvascular networks.
- [`testcase_distensibility.py`](https://github.com/Franculino/microBlooM/blob/main/testcases/testcase_distensibility.py): stationary blood flow in microvascular networks considering vascular distensibility, i.e., the ability of blood vessels to passively change their diameters with respect to intra- and extravascular pressure.
- [`testcase_inverse_problem.py`](https://github.com/Franculino/microBlooM/blob/main/testcases/testcase_inverse_problem.py): an inverse model approach for estimating vascular parameters such as diameters and transmissibilities of microvascular networks based on given flow rates and velocities in selected vessels.
- [`testcase_bc_tuning.py`](https://github.com/Franculino/microBlooM/blob/main/testcases/testcase_bc_tuning.py): an inverse model approach for estimating network boundary conditions based on given flow rates and velocities in selected vessels.

Please find a more detailed description for each test case in the corresponding test cases file.

**NOTE: all parameters are in S.I. units**

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.10
- Numpy 1.23.4
- Matplotlib 3.6.2
- Scipy 1.9.3
- igraph 0.10.2
- Pandas 1.5.1
- Pyamg 4.2.3

## Installation

1. Clone the repository:

```
git clone https://github.com/Franculino/microBlooM.git
```

2. Move to the correct directoy

```
cd microBlooM
```

3. Run the [`main.py`](https://github.com/Franculino/microBlooM/blob/main/main.py) file

```
python3 main.py
```

The framework does not have yet an executable file to launch the program and select the desired outcome, refer to [Usage](#usage) for further information.

## Usage

### Run the simulation

The available simulation can be run from [`main.py`](https://github.com/Franculino/microBlooM/blob/main/main.py). In order to select the desired simulation it is necessary to uncomment the specific test cases.

### Input files

A prerequisite to run the simulation is the vascular network in graph format. The graph can be loaded in igraph format either stored in pickle file (.pkl) or CSV, that need to be stored in `data\network` folder and modify the relative path in the chosen test case file. In case there is no network available, it is possible to create a hexagonal network.

The specific format for both cases is detailed in [`fileio\read_netwowrk.py`](https://github.com/Franculino/microBlooM/blob/main/source/fileio/read_network.py) and below.

<details>
 <summary>  CSV </summary>

Import a network from three csv text files containing vertex, edge and boundary data.

_Vertex (vx) data_: At least three columns are required to describe the x, y and z coordinates of all vertices. A
header for each column has to be provided. Example file structure (order of columns does not matter; columns without header are ignored):

        x_coord, y_coord, z_coord
        x_coord_of_vx_0,y_coord_of_vx_0,z_coord_of_vx_0
        x_coord_of_vx_1,y_coord_of_vx_1,z_coord_of_vx_1
                :      ,        :      ,        :
                :      ,        :      ,        :

_Edge data_: At least four columns are required to describe the connectivity (requires two columns, i.e.,
one for each incidence vertex per edge), diameters and lengths of all edges. A header for each column has to be
provided. Example file structure (order of columns does not matter; columns without header are ignored):

        vx_incident_1, vertex_incident_2, diameter, length
        incident_vx_1_of_edge_0,incident_vx_2_of_edge_0,diameter_of_edge_0,length_of_edge_0
        incident_vx_1_of_edge_1,incident_vx_2_of_edge_1,diameter_of_edge_1,length_of_edge_1
                    :          ,            :          ,            :     ,         :
                    :          ,            :          ,            :     ,         :

_Boundary data_: At least three columns are required to prescribe the vertex indices of boundary conditions,
the boundary type (1: pressure, 2: flow rate) and the boundary values (can be pressure or flow rate).
Example file structure (order of columns does not matter; columns without header are ignored):

        vx_id_of_boundary, boundary_type, boundary_value
        vx_id_boundary_0,boundary_type_0,boundary_value_0
        vx_id_boundary_1,boundary_type_1,boundary_value_1
                :       ,       :       ,       :
                :       ,       :       ,       :

</details>

<details>
 <summary>  Igraph (.pkl) </summary>

 The loaded igraph needs to contain the following vertex and edge attributes. The name of the respective attributes can be modified in the test case files. The default names are stated below. Note the boundary data is also stored as vertex attribute, i.e. in total there are at least three vertex attributes.

_Vertex data_: At least one attribute is required to describe the x, y and z coordinates of all vertices ("coords").

```
one (3 x nv) array, where nv is the number of vertices:
        [[x0, y0, z0]
         [x1, y1, z1]
                    :
                    :
         [xnv, ynv, znv]]
```

_Edge data_: At least two attributes are required to describe the diameters ("diameter") and lengths ("length") of all edges.

```
two (1 x ne) arrays, where ne is number of edges:
        diameter: [d0, d1, ..., dne ]
        length: [l0, l1, ..., lne ]
```

_Boundary data_: At least two vertex attributes are required to prescribe the boundary type (1: pressure, 2: flow rate, None: otherwise, "boundaryType") and the boundary values (can be pressure or flow rate, None: otherwise, "boundaryValue").

```
two (1 x nv) arrays, where nv is number of vertices:
            boundary_type: [boundary_type_0, boundary_type_1, ..., boundary_type_nv]
            boundary_value: [boundary_value_0, boundary_value_1, ..., boundary_value_nv]
```

</details>

<details>

 <summary>  Hexagonal Network </summary>

The hexagonal network properties can be modified from the `testcase` file of the choose simulation. Here an example of possible values:

        "nr_of_hexagon_x": 3,
        "nr_of_hexagon_y": 3,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 27],
        "hexa_boundary_values": [2, 1],
        "hexa_boundary_types": [1, 1],

Note: the number odd hexagon must be odd.

</details>

### Output Network files

If switched on in the test case file, the simulation output can be saved.
The possible formats are igraph format (in a `.pkl` file), `vtp` format or `CSV` file. It is necessary to set the relative path with the desired output folder in the test case file.


## Contributing

**microBlooM** has been developed by Franca Schmid (FS), Robert Epp (RE) and Chryso Lambride (CL).

Please cite the repository and the following papers when using the blood flow model [1] and when using the inverse model [2,3,4] (See Bibliography).


## LICENCE

This project is licensed under the terms of the [GNU General Public License v3.0](https://github.com/Franculino/microBlooM/blob/main/LICENSE)


## Contact

Please contact FS in case of questions or requests (franca.schmid@unibe.ch).


## Bibliography

[1] Schmid, F., Tsai, P. S., Kleinfeld, D., Jenny, P., & Weber, B. (2017). [Depth-dependent flow and pressure characteristics in cortical microvascular networks](https://doi.org/10.1371/journal.pcbi.1005392). PLoS Computational Biology, 13(2), e1005392.

[2] Epp, R., Schmid, F., Weber, B., Jenny, P. (2020). [Predicting vessel diameter changes to up-regulate bi-phasic blood flow during activation in realistic microvascular networks.](https://doi.org/10.3389/fphys.2020.566303) Frontiers in Physiology, 11, 1132.

[3] Epp, R., Glück, C., Binder, N.F,, El Amki, M., Weber, B., Wegener, S., Jenny, P., Schmid, F., 2023. [The role of leptomeningeal collaterals in redistributing blood flow during stroke](https://doi.org/10.1371/journal.pcbi.1011496). PLoS Computational Biology.

[4] Epp, R., Schmid, F. , Jenny, P., 2022. [Hierarchical regularization of solution ambiguity in underdetermined inverse optimization problems](https://doi.org/10.1016/j.jcpx.2022.100105). Journal of Computational Physics: X, 13(100105).

[5] Pries, A. R., Neuhaus, D., & Gaehtgens, P. (1992). [Blood viscosity in tube flow: dependence on diameter and hematocrit](https://doi.org/10.1152/ajpheart.1992.263.6.H1770). The American journal of physiology, 263(6 Pt 2), H1770–H1778. 

[6] Pries, A. R., & Secomb, T. W. (2005). [Microvascular blood viscosity in vivo and the endothelial surface layer](https://doi.org/10.1152/ajpheart.00297.2005). American journal of physiology. Heart and circulatory physiology, 289(6), H2657–H2664. 

[7] Sherwin, S.J., Franke, V., Peiro, J., Parker, K., 2003. [One-dimensional modelling of a vascular network in space- time variables](https://doi.org/10.1023/B:ENGI.0000007979.32871.e2). Journal of Engineering Mathematics 47(3), 217-250.
