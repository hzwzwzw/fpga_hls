# Create a new project for the MPE
open_project mpe_prj

# Set the top-level function for synthesis to mpe_mv
set_top mpe_mv

# Add all necessary source files. MPE depends on VPU.
add_files src/mpe.cpp
add_files src/mpe.hpp
add_files src/vpu.cpp
add_files src/vpu.hpp

# Add the testbench. HLS will use it to verify the mpe_mv function.
add_files -tb test/mpe_test.cpp

# Create a solution
open_solution "solution1"

# Set the target device and clock
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 4.44 -name default

# Run C synthesis
csynth_design

# Exit Vitis HLS
exit
