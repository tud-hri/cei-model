# Simple Merging Simulation for CEI model
This project contains a simulation environment for two vehicle in a simplified merging scenario. The scenario is symmetrical and the vehicles are modelled as point mass objects that can move in a single dimension. The acceleration of these vehicles can be controlled by computational agents. This simulation environment was created to study the Communication-Enabled Interaction Model (CEI Model). 
 

## CEI Model
The CEI model framework is a model framework for traffic interactions that explicitly accounts for the interaction between traffic participants. Besides that, it does not assume that humans are rational utility maximizers. Instead, it assumes that humans initialize behavior and keep on executing their current plan until their perceived risk either exceeds an upper threshold or is below a lower threshold for an extended amount of time. Another key aspect of the model is that humans are assumed to communicate their plan to the other vehicle because it is in both drivers interest to prevent collisions.

A full description of this implementation of a CEI model can be found in the paper "A communication-enabled modelling framework to describe reciprocal interactions between two drivers". 

## Run instructions
To run the simulations that are reported in the paper, see the run script in the module `run_scenarios`. The plots reported in the paper can then be reproduced using the scripts in the `plotting` module. All simulated results can also be replayed and visualized by running the script `playback.py` from the main folder. All data is stored in a `data` folder.  
 
## General structure
The simulation in this package is controlled by a `simmaster` object. This object controls the timing and calls the updates on agents, objects, and the GUI. There are multiple (e.g., for offline simulation and real-time playback). You can find these in the `simulation` module. The simulation uses a `Track` for the vehicles to drive on, these can be found in `trackobjects`. Each vehicle is represented by a `PointMassObject` from the module `controllableobjects`, and it is controlled by an `agent` from the `agents` module. The other modules contain tests, tools, run, and plotting scripts.

If you have more specific questions, please don't hesitate to open an issue or reach out to me.
