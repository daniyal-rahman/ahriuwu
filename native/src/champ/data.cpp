#include "data.hpp"

#include <stdexcept>
#include <unordered_map>

namespace lanesim::data {

namespace {
std::unordered_map<std::string, std::vector<float>>& store() {
    static std::unordered_map<std::string, std::vector<float>> s;
    return s;
}
}  // namespace

void put(const std::string& key, const float* v, size_t n) { store()[key] = std::vector<float>(v, v + n); }

const std::vector<float>& table(const std::string& key) {
    auto it = store().find(key);
    if (it == store().end()) throw std::out_of_range("lanesim data: no constant '" + key + "' (native/python/consts)");
    return it->second;
}

}  // namespace lanesim::data

extern "C" void ls_data_put(const char* key, const float* v, long n) { lanesim::data::put(key, v, (size_t)n); }
