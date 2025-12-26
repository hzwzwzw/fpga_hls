#ifndef CONFIG_HPP
#define CONFIG_HPP

#include "mpe.hpp"

// Define the tile dimensions. For our architecture, the tile size
// must match the MPE size, as the MPE is our core processing block.
#define TILE_ROWS MPE_ROWS
#define TILE_COLS MPE_COLS

// For now, we assume a single MPE. We will explore NUM_MPES later.
#define NUM_MPES 1

#endif // CONFIG_HPP
