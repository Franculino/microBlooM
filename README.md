# microBlooM

**MicroBloomM** is an open-source simulation framework that has been generated to improve our understanding of the flow characteristics in a microvasculature. The numerical simulations computate the blood flow in microvascular networks, considering the impact of red blood cells (Fåhraeus-Linquist effects). The elasticity of the blood vessels has been included, allowing the simulation of passive vascular diameter adaptations with respect to pressure changes. 

Furthermore a novel *inverse model* has been developed for microvascular blood flow that is capable of inferring vascular parameters such as vascular diameter and transmissibility based on prescribed flow characteristics. In addition, the inverse model has been extended to predict the network boundary conditions that are required to obtain the desired flow characteristics.


 
The simulations are associated with test cases that can be modified by the user (see [Usage](#usage)). The following list reports the designed test:

-	[`testcase_blood_flow_model.py`](https://github.com/Franculino/microBlooM/blob/main/testcases/testcase_blood_flow_model.py): stationary blood flow in microvascular networks.
-	[`testcase_distensibility.py`](https://github.com/Franculino/microBlooM/blob/main/testcases/testcase_distensibility.py): stationary blood flow in microvascular networks considering vascular distensibility, i.e., the ability of blood vessels to passively change their diameters with respect to intra- and extravascular pressure.
-	[`testcase_inverse_problem.py`](https://github.com/Franculino/microBlooM/blob/main/testcases/testcase_inverse_problem.py): an inverse modelling approach for estimating vascular parameters such as diameters and transmissibilities of microvascular networks based on given flow rates and velocities in selected vessels.
-	[`testcase_bc_tuning.py`](https://github.com/Franculino/microBlooM/blob/main/testcases/testcase_bc_tuning.py): an inverse modelling approach for estimating network boundary conditions based on given flow rates and velocities in selected vessels.

Please find a more detailed description for each test case in the corresponding testcases file.
 

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

The framework does not have yet an executable file to launch the program and select the desire outcome, refer to [Usage](#usage) for further information.



## Usage
### Run the simulation

The available simulation can be run from [`main.py`](https://github.com/Franculino/microBlooM/blob/main/main.py) and in order to select the desired simulation it is necessary to uncomment the specific test cases. 
### Input files

The framework could accept as input files: graph in igraph format (stored in pickle file, `.pkl`) or `CSV` files, that need to be store in `data\network` folder and modify the relative path in the choosen test case file. In case there are no network available it is possible to create a hexagonal network. 

The specific format for both cases is detailed in [`fileio\read_netwowrk.py`](https://github.com/Franculino/microBlooM/blob/main/source/fileio/read_network.py) and below.


<details>
 <summary>  CSV </summary>

Import a network from the three csv text files containing vertex, edge and boundary data.

*Vertex (vx) data*: At least three columns are required to describe the x, y and z coordinates of all vertices. A
        header for each column has to be provided. Example file structure (order of columns does not matter; additional
        columns are ignored):

        x_coord, y_coord, z_coord
        x_coord_of_vx_0,y_coord_of_vx_0,z_coord_of_vx_0
        x_coord_of_vx_1,y_coord_of_vx_1,z_coord_of_vx_1
                :      ,        :      ,        :
                :      ,        :      ,        :

*Edge data*: At least four columns are required to describe the incidence vertices (requires two columns, i.e.,
        one for each incidence vertex per edge), diameters and lengths of all edges. A header for each column has to be
        provided. Example file structure (order of columns does not matter; additional columns are ignored):

        vx_incident_1, vertex_incident_2, diameter, length
        incident_vx_1_of_edge_0,incident_vx_2_of_edge_0,diameter_of_edge_0,length_of_edge_0
        incident_vx_1_of_edge_1,incident_vx_2_of_edge_1,diameter_of_edge_1,length_of_edge_1
                    :          ,            :          ,            :     ,         :
                    :          ,            :          ,            :     ,         :

*Boundary data*: At least three columns are required to prescribe the vertex indices of boundary conditions,
        the boundary type (1: pressure, 2: flow rate) and the boundary values (can be pressure or flow rate).
        Example file structure (order of columns does not matter; additional columns are ignored):

        vx_id_of_boundary, boundary_type, boundary_value
        vx_id_boundary_0,boundary_type_0,boundary_value_0
        vx_id_boundary_1,boundary_type_1,boundary_value_1
                :       ,       :       ,       :
                :       ,       :       ,       :

</details>

<details>
 <summary>  Igraph (.pkl) </summary>


 *Vertex data*: At least one attribute is required to describe the x, y and z coordinates of all vertices.
```
one (3 x nv) array, where nv is the number of vertices:
        [[x0, y0, z0]
         [x1, y1, z1]
                    :
                    :
         [xnv, ynv, znv]]
```
*Edge data*: At least two attribute are required to describe the diameters and lengths of all edges.
```
two (1 x ne) arrays, where ne is number of edges:
        diameter: [d0, d1, ..., dne ]
        length: [l0, l1, ..., lne ]
```
*Boundary data*: At least two vertex attributes are required to prescribe the boundary type (1: pressure, 2: flow rate, None: otherwise) and the boundary values (can be pressure or flow rate,
None: otherwise).
```
two (1 x nv) arrays, where nv is number of vertices:
            boundary_type: [boundary_type_0, boundary_type_1, ..., boundary_type_nv]
            boundary_value: [boundary_value_0, boundary_value_1, ..., boundary_value_nv]
```
</details>




<details>
 <summary>  Hexagonal Network </summary>

 The hexagonal network properties can be modify from the `testcase` file of the choose simulation.  Here an example of possible values:

        "nr_of_hexagon_x": 3,
        "nr_of_hexagon_y": 3,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 27],
        "hexa_boundary_values": [2, 1],
        "hexa_boundary_types": [1, 1],
Note: the number od hexagon must be odd.
</details>

### Output Network files
The simulation can provide as output the resulting network if the parameter has been set in the test case file.
The possible formats are igraph format (in a `.pkl` file), `vtp` format or `CSV` file. It is necessary to set the relative path with the desired output folder in the test case file.


### Test cases
The simulation can be fitted for a specific network by the user that can modify the following parameter:






## Contributing
microBlooM has been developed by Franca Schmid (FS), Robert Epp (RE) and Chryso Lambride (CL). 

Please cite the repository and the following papers when using it:

[1] Schmid, F., Tsai, P. S., Kleinfeld, D., Jenny, P., & Weber, B. (2017). Depth-dependent flow and pressure characteristics in cortical microvascular networks. PLoS Computational Biology, 13(2), e1005392.

[2] Epp, R., Schmid, F., Weber, B., Jenny, P. (2020). Predicting vessel diameter changes to up-regulate bi-phasic blood flow during activation in realistic microvascular networks. Frontiers in Physiology, 11, 1132.  
## LICENCE

This project is licensed under the terms of the [GNU General Public License v3.0](https://github.com/Franculino/microBlooM/blob/main/LICENSE)
