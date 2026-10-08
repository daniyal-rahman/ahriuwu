// Named-section profiler for finding hot spots below the phase level: LS_PROF("name") times the enclosing scope
// into a per-thread section (one lookup per call site and thread). Off unless enabled (ls_prof_enable): a disabled
// scope costs one relaxed load.
#pragma once
#include <atomic>
#include <chrono>
#include <string>
#include <vector>

namespace lanesim::prof {

struct Section {
    const char* name;
    double ns = 0;
    long calls = 0;
};

extern std::atomic<bool> enabled;
Section* section(const char* name);                 // this thread's section for ``name``
std::string dump(bool reset);                       // "name ns calls\n" for this thread's sections

struct Scope {
    Section* s = nullptr;
    std::chrono::steady_clock::time_point t;
    explicit Scope(Section* sec) {
        if (sec) s = sec, t = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (s) s->ns += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t).count(), ++s->calls;
    }
};

// Consecutive sections of one function: LS_LAP(laps, "name") charges the time since the previous lap.
struct Laps {
    bool on = enabled.load(std::memory_order_relaxed);
    std::chrono::steady_clock::time_point t = on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    void lap(Section*& sec, const char* name) {
        if (!on) return;
        if (!sec) sec = section(name);
        auto now = std::chrono::steady_clock::now();
        sec->ns += std::chrono::duration<double, std::nano>(now - t).count(), ++sec->calls;
        t = now;
    }
};

}  // namespace lanesim::prof

#define LS_PROF_CAT2(a, b) a##b
#define LS_PROF_CAT(a, b) LS_PROF_CAT2(a, b)
#define LS_PROF(name)                                                                                         \
    static thread_local ::lanesim::prof::Section* LS_PROF_CAT(ls_prof_sec_, __LINE__) = nullptr;              \
    ::lanesim::prof::Scope LS_PROF_CAT(ls_prof_scope_, __LINE__)(                                             \
        ::lanesim::prof::enabled.load(std::memory_order_relaxed)                                               \
            ? (LS_PROF_CAT(ls_prof_sec_, __LINE__) ? LS_PROF_CAT(ls_prof_sec_, __LINE__)                       \
                                                   : (LS_PROF_CAT(ls_prof_sec_, __LINE__) = ::lanesim::prof::section(name))) \
            : nullptr)
#define LS_LAP(laps, name)                                                                                    \
    do {                                                                                                      \
        static thread_local ::lanesim::prof::Section* ls_lap_sec = nullptr;                                   \
        (laps).lap(ls_lap_sec, name);                                                                         \
    } while (0)
