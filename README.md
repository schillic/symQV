# symQV

## Symbolic Quantum Program Verification

This Python project comes with the experiments

- Toffoli Gate
- Quantum Teleportation Protocol
- 8-Bit Adder
- Quantum Fourier Transform
- Quantum Phase Estimation
- Grover Diffusion Operator

that show how symQV can prove correctness of quantum algorithms.

## Installation

There are three requirements for symQV to run. The Python packages are:

- z3-solver (for SMT sort declaration and syntax generation)
- pyparsing (needed for output parsing)
- numpy (for arithmetic)
- scipy (for optimization)

Also, the system symQV runs on requires [dReal](http://dreal.github.io) to be installed and added to the path
in order for the solver to work.

Install Python packages:

    pip install z3-solver
    pip install pyparsing
    pip install numpy
    pip install scipy

Install dReal on Mac:

    /usr/bin/curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/mac/install.sh | bash
    dreal

Install dReal on Ubuntu:

    # 20.04
    sudo apt-get install curl
    curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/20.04/install.sh | sudo bash
    
    # 18.04
    sudo apt-get install curl
    curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/18.04/install.sh | sudo bash

Alternatively, make the shell script `install.sh` executable and run it:

    sudo chmod +x install.sh
    ./install.sh

## Examples

The code to generate all figures and analysis is available with the repeatability
package.

Examples to reproduce the data are included in the top-level directory, including a script `run.sh` to run all examples.
This is a simple helper script which runs the following commands:

    python experiment_toffoli.py
    python experiment_teleportation.py
    python experiment_add8.py
    python experiment_qft.py
    python experiment_qpe.py
    python experiment_gdo.py

__IMPORTANT__: You may need to add execution privileges to the script in order to run it on your machine.
Also, please note that the examples are provided in order to show the functionality of the tool,
but due to memory constraints, we have provided a "stripped down" version of the examples to showcase the
functionality of the tool.