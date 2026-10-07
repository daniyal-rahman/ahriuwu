// Load/store a generated value type from/to its run of Env fields. A ModernState subtree (s.combat, s.champ,
// s.kits, ...) is a contiguous run of leaves in flatten order, the order the value type's visit() walks.
#pragma once
#include <cstring>
#include <string>

#include "../world.hpp"
#include "marshal.hpp"

namespace lanesim {

long env_index(const char* name);                  // index of the Env field (api.cpp)

template <class T>
void env_load(const World& w, const Env& e, long first, T& out) {
    void* const* ptrs = reinterpret_cast<void* const*>(&e) + first;
    std::vector<long> counts(w.env_counts.begin() + first, w.env_counts.end());
    marshal::LoadVisitor lv{ptrs, counts.data(), 0};
    lv(out);
}

struct EnvStoreVisitor {
    void* const* ptrs;
    size_t i = 0;
    template <class U> void operator()(U& m) {
        if constexpr (std::is_arithmetic_v<U>) {
            std::memcpy(ptrs[i++], &m, sizeof(U));
        } else if constexpr (marshal::has_visit<U>::value) {
            m.visit(*this);
        } else {
            using E = std::remove_reference_t<decltype(m[0])>;
            if (m.size()) std::memcpy(ptrs[i], m.data(), m.size() * sizeof(E));
            ++i;
        }
    }
};

template <class T>
void env_store(const Env& e, long first, T& v) {
    EnvStoreVisitor sv{reinterpret_cast<void* const*>(&e) + first, 0};
    sv(v);
}

}  // namespace lanesim
