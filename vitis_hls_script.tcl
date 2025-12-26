# Create a new project
open_project vpu_prj

# Set the top-level function for synthesis
set_top vpu

# Add source and testbench files
add_files src/vpu.cpp
add_files src/vpu.hpp
add_files -tb test/vpu_test.cpp

# Create a solution
open_solution "solution1"

# Set the target device (Alveo U280, as mentioned in the paper)
# The part number is for the U280 card
set_part {xcu280-fsvh2892-2L-e}

# Set the target clock period to 4.44ns, which corresponds to 225MHz (as in the paper)
create_clock -period 4.44 -name default

# Run C simulation
csim_design

# Run C synthesis
csynth_design

# Run C/RTL co-simulation for verification
# cosim_design

# Exit Vitis HLS
exit
