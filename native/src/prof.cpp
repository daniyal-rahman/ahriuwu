#include "prof.hpp"

#include "champ/arr.hpp"

#include <cstring>
#include <deque>

namespace lanesim::prof {

std::atomic<bool> enabled{false};

namespace {
thread_local std::deque<Section> sections;          // stable addresses
}

Section* section(const char* name) {
    for (Section& s : sections)
        if (std::strcmp(s.name, name) == 0) return &s;
    sections.push_back({name});
    return &sections.back();
}

std::string dump(bool reset) {
    std::string out = "arr.heap_allocs 0 " + std::to_string(arr_heap_allocs()) + "\n";
    if (reset) arr_heap_allocs() = 0;
    for (Section& s : sections) {
        out += std::string(s.name) + " " + std::to_string(s.ns) + " " + std::to_string(s.calls) + "\n";
        if (reset) s.ns = 0, s.calls = 0;
    }
    return out;
}

}  // namespace lanesim::prof
