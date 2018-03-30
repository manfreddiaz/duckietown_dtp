# Detection, Tracking and Prediction in Duckietown Environment


## TL;DR

This repository contains preliminary/protoypical results for the Detection, Tracking and Prediction research in the Duckietown environment.
For a detailed explanantion of the goals and final results of this project, please consult the Duckietown [book]() / [raw](https://github.com/duckietown/duckuments/blob/devel-implicit-coord/docs/atoms_85_fall2017_projects/19_impicit-coord/30-udem-implicit-coord-theory.md). 
There are also [slides]
(https://docs.google.com/presentation/d/1dD_olppPbkYwsf1wybxyhrnvNNKolLpeyMU4JjcoRA4/edit#slide=id.gc6f73a04f_0_9) available with demos/visualizations that could help to gain a better understanding.
We also created visualizations that are available [here](https://www.youtube.com/playlist?list=PLkWqVLQ4U20QCW3gXhFBJk1cpwWiD7myZ). 

## Authors
- Manfred Diaz ([github](https://github.com/takeitallsource))
- Jonathan Arsenault ([github](https://github.com/jonarsenault))

## Motivation

Implicit coordination of traffic at an intersection comprises the orchestration without any form of explicit communication of the entities involved, such as traffic lights or signs, in-vehicle signalization or vehicle-to-vehicle (to-infrastructure) communication systems. Thus, the outcome of such mechanism is to produce an accurate inference of when it is safe to progress with a crossing maneuver.

As of today, Duckietown exhibits a less complex environment -compared to real-life situations- where the only mobile entities are duckiebots. This simplification provides a favorable scenario to explore techniques at different levels of complexity which could be incrementally built to produce algorithms and heuristics applicable to more convoluted scenarios.

Predicting traffic behavior at an intersection depends on accurately detect and track the position of each object as the preamble of applying prior information (traffic rules) for predicting the sequence of expected actions of each element. Hence, the conception of a mechanism that implicitly coordinates the individual behavior of a duckiebot under such circumstances comprises the research, design, and implementation of components capable of producing the required data for this outcome.

## Object Detection

[doc](https://github.com/duckietown/duckuments/blob/devel-implicit-coord/docs/atoms_85_fall2017_projects/19_impicit-coord/30-udem-implicit-coord-theory.md#detection) / [code](https://github.com/duckietown/Software/tree/devel_implicit_coord/catkin_ws/src/80-deep-learning/object_detection)

## Multi-Vehicle Tracker

[doc](https://github.com/duckietown/duckuments/blob/devel-implicit-coord/docs/atoms_85_fall2017_projects/19_impicit-coord/30-udem-implicit-coord-theory.md#tracking) / [code](https://github.com/duckietown/Software/tree/devel_implicit_coord/catkin_ws/src/50-misc-additional-functionality/multivehicle_tracker)

## Debugging / Visualization Tools

[code](https://github.com/duckietown/Software/tree/devel_implicit_coord/catkin_ws/src/70-convenience-packages/multivehicle_tracking_visualizer)
