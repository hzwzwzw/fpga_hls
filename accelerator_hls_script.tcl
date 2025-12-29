# Create a new project for the accelerator
open_project accelerator_prj

# Set the top-level function for synthesis
set_top flight_llm_accelerator

# Add all source files required for the accelerator
add_files src/accelerator.cpp
add_files src/mpe.cpp
add_files src/vpu.cpp
# Add headers (optional for HLS but good practice)
add_files src/accelerator.hpp
add_files src/config.hpp
add_files src/mpe.hpp
add_files src/vpu.hpp

# Add the testbench
add_files -tb test/accelerator_test.cpp

# Create a solution
open_solution "solution1"

# Set the target device and clock
set_part {xcu280-fsvh2892-2L-e}
create_clock -period 4.44 -name default

# Run C synthesis
csynth_design

# Exit Vitis HLS
exit
