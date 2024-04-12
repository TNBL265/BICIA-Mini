# BIC Mini Project

## How to Run:
- The initial tutorial for MNIST can be found here: https://www.nengo.ai/nengo-fpga/examples.html#mnist-digit-classifier
- I have rewrote the code for my experiments, removal off all FPGA board related since I don't have one. (SNN might show actually shows better accuracy if I have an FPGA board to play with, my manual parameters tuning was not enough).
- Install Python package `pip install -r requirements.txt`
- Both the simulation and the Nengo GUI import the model from `model_def.py`
  - to run GUI: `nengo gui.py`
  - to run simulation: `python sim.py`
