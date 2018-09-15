Mushroom Hierarchical
*********************

**Mushroom Hierarchical: Hierarchical Reinforcement Learning python library.**

This repository implements the hierarchical framework proposed in the paper: "A Control Theoretic Approach to Hierarchical Reinforcement Learning"

What is Mushroom Hierarchical
=============================
Mushroom Hierarchical is a python Hierarchical Reinforcement Learning (HRL) library based on Mushroom, the RL library of Politecnico di Milano. 
It allows to perform HRL experiments exploiting the control graph formalism.


Installation
============

You can do a minimal installation of ``Mushroom`` with:

.. code:: shell

	git clone https://github.com/AIRLab-POLIMI/mushroom.git
	cd mushroom
	pip3 install -e .

You can install ``Mushroom Hierarchical`` with:

.. code:: shell

	git clone https://github.com/AIRLab-POLIMI/mushroom_hierarchical.git
	cd mushroom_hierarchical
	pip3 install -e .

How to set and run and experiment
=================================
To run experiments, you should use the run.py  script in the experiment folder.
If you want to visualize the results produced by our runs, you can look at the out folder, or you can re-run the plot_*.py scripts into the graphs folder.
