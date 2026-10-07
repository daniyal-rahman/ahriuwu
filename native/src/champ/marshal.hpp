// Generic marshalling between flattened JAX pytrees (numpy leaves) and the generated value types, and a registry
// of native functions callable from Python with JAX arguments (hook differential tests,
// ops/native/test_hooks.py).
//
// A value type's ``visit`` enumerates members in JAX flatten order; scalars are one leaf, ``Arr`` one flat leaf,
// nested types recurse. Leaves are typed by a code: 'f' float32, 'i' int32, 'u' uint32, 'b' bool (uint8).
#pragma once
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "arr.hpp"

namespace lanesim::marshal {

template <class T> struct code_of;
template <> struct code_of<float> { static constexpr char v = 'f'; };
template <> struct code_of<int32_t> { static constexpr char v = 'i'; };
template <> struct code_of<uint32_t> { static constexpr char v = 'u'; };
template <> struct code_of<uint8_t> { static constexpr char v = 'b'; };

struct AnyVisitor {
    template <class U> void operator()(U&) {}
};
template <class T, class = void> struct has_visit : std::false_type {};
template <class T>
struct has_visit<T, std::void_t<decltype(std::declval<T&>().visit(AnyVisitor{}))>> : std::true_type {};

// One flat leaf: element type code, element count, data.
struct Leaf {
    char code;
    std::vector<uint8_t> bytes;
    long count;
};

// Leaf type codes of a default-constructed T (the signature Python converts its arguments to).
template <class T> void signature(T& v, std::string& out);

struct SigVisitor {
    std::string* out;
    template <class U> void operator()(U& m) {
        if constexpr (std::is_arithmetic_v<U>) out->push_back(code_of<U>::v);
        else if constexpr (has_visit<U>::value) m.visit(*this);
        else out->push_back(code_of<std::remove_reference_t<decltype(m[0])>>::v);
    }
};

template <class T> void signature(T& v, std::string& out) {
    SigVisitor s{&out};
    s(v);
}

struct LoadVisitor {
    void* const* ptrs;
    const long* counts;
    size_t i = 0;
    template <class U> void operator()(U& m) {
        if constexpr (std::is_arithmetic_v<U>) {
            std::memcpy(&m, ptrs[i++], sizeof(U));
        } else if constexpr (has_visit<U>::value) {
            m.visit(*this);
        } else {
            using E = std::remove_reference_t<decltype(m[0])>;
            m.resize(counts[i]);
            if (counts[i]) std::memcpy(m.data(), ptrs[i], counts[i] * sizeof(E));
            ++i;
        }
    }
};

struct StoreVisitor {
    std::vector<Leaf>* out;
    template <class U> void operator()(U& m) {
        if constexpr (std::is_arithmetic_v<U>) {
            Leaf l{code_of<U>::v, std::vector<uint8_t>(sizeof(U)), 1};
            std::memcpy(l.bytes.data(), &m, sizeof(U));
            out->push_back(std::move(l));
        } else if constexpr (has_visit<U>::value) {
            m.visit(*this);
        } else {
            using E = std::remove_reference_t<decltype(m[0])>;
            Leaf l{code_of<E>::v, std::vector<uint8_t>(m.size() * sizeof(E)), (long)m.size()};
            if (m.size()) std::memcpy(l.bytes.data(), m.data(), m.size() * sizeof(E));
            out->push_back(std::move(l));
        }
    }
};

// A registered native function: argument signature, and a call from flat leaves to flat output leaves.
struct Entry {
    std::string signature;      // leaf codes of all arguments, in order
    std::vector<int> arg_leaves;  // leaf count per argument
    std::function<std::vector<Leaf>(void* const*, const long*)> call;
};

inline std::map<std::string, Entry>& registry() {
    static std::map<std::string, Entry> r;
    return r;
}

template <class T> void store_any(T& v, std::vector<Leaf>& out) {
    StoreVisitor s{&out};
    s(v);
}
template <class... T> void store_any(std::tuple<T...>& v, std::vector<Leaf>& out) {
    std::apply([&](auto&... x) { (store_any(x, out), ...); }, v);
}

// Register ``fn(args...) -> result`` (result: a value type or a std::tuple of them) under ``name``. Arguments
// are taken by value (deduced from the function signature, references stripped).
template <class R, class... A>
bool add(const std::string& name, R (*fn)(A...)) {
    Entry e;
    std::tuple<std::decay_t<A>...> probe{};
    std::apply([&](auto&... x) {
        (([&] {
             std::string s;
             signature(x, s);
             e.signature += s;
             e.arg_leaves.push_back((int)s.size());
         }()), ...);
    }, probe);
    e.call = [fn](void* const* ptrs, const long* counts) {
        std::tuple<std::decay_t<A>...> args{};
        LoadVisitor lv{ptrs, counts, 0};
        std::apply([&](auto&... x) { (lv(x), ...); }, args);
        auto result = std::apply([&](auto&... x) { return fn(x...); }, args);
        std::vector<Leaf> out;
        store_any(result, out);
        return out;
    };
    registry()[name] = std::move(e);
    return true;
}

}  // namespace lanesim::marshal

// At namespace scope in a .cpp: LANESIM_TEST(items_fighter_on_hit, "items.fighter.on_hit", fighter::on_hit);
#define LANESIM_TEST(id, name, fn) static const bool lanesim_test_##id = ::lanesim::marshal::add(name, &fn)
