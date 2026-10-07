// Named constants and tables from the Python side (native/python/consts/*.py), evaluated with the same helpers
// the JAX modules use (catalog dv, rune ea, champion values/cooldowns, economy tables). Loaded once per process
// before any tick (lanesim.load_consts); modules read them into function-local statics.
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace lanesim::data {

void put(const std::string& key, const float* v, size_t n);
const std::vector<float>& table(const std::string& key);   // throws std::out_of_range naming the key
inline float f(const std::string& key) { return table(key).at(0); }

}  // namespace lanesim::data
