// The full 26.19 top-lane tick with champions (world/tick.py).
#pragma once
#include "../world.hpp"

namespace lanesim::champ {

TickStats step_full(const World& w, Env& e, Orders& o);

}  // namespace lanesim::champ
