// Arr<T>: a small vector with inline storage for short arrays (the (C,) per-champion fields), heap beyond.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace lanesim {

// Heap blocks of Arr: per-thread free lists of power-of-two sizes. A tick allocates and frees thousands of
// short-lived arrays of a few sizes; recycling them skips the allocator. Blocks never return to the system (the
// pool's high-water mark stays small); a block freed on another thread joins that thread's list.
namespace arr_pool {
constexpr int MIN_CLASS = 4, CLASSES = 40;           // 16-byte minimum block
struct Lists {
    void* head[CLASSES] = {};
    long fresh = 0;                                     // blocks taken from the system (prof "arr.heap_allocs")
};
inline Lists& lists() {
    static thread_local Lists l;
    return l;
}
inline int size_class(size_t bytes) {
    int k = MIN_CLASS;
    while ((size_t(1) << k) < bytes) ++k;
    return k;
}
inline void* get(int k) {
    Lists& l = lists();
    if (void* p = l.head[k]) {
        l.head[k] = *static_cast<void**>(p);
        return p;
    }
    ++l.fresh;
    return ::operator new(size_t(1) << k);
}
inline void put(void* p, int k) {
    Lists& l = lists();
    *static_cast<void**>(p) = l.head[k];
    l.head[k] = p;
}
}  // namespace arr_pool

inline long& arr_heap_allocs() { return arr_pool::lists().fresh; }

template <class T, size_t Inline = 4>
class Arr {
    static_assert(std::is_trivially_copyable_v<T>);
    T small_[Inline];
    T* data_ = small_;
    size_t size_ = 0, cap_ = Inline;

  public:
    Arr() = default;
    explicit Arr(size_t n, T fill = T()) { assign(n, fill); }
    Arr(const Arr& o) { copy(o); }
    Arr& operator=(const Arr& o) {
        if (this != &o) copy(o);
        return *this;
    }
    Arr(Arr&& o) noexcept { take(o); }
    Arr& operator=(Arr&& o) noexcept {
        if (this != &o) {
            release();
            data_ = small_, cap_ = Inline;
            take(o);
        }
        return *this;
    }
    ~Arr() { release(); }
    void reserve(size_t n) {
        if (n <= cap_) return;
        int k = arr_pool::size_class(n * sizeof(T));
        T* d = static_cast<T*>(arr_pool::get(k));
        std::memcpy(d, data_, size_ * sizeof(T));
        release();
        data_ = d, cap_ = (size_t(1) << k) / sizeof(T);
    }
    void resize(size_t n, T fill = T()) {
        reserve(n);
        for (size_t i = size_; i < n; ++i) data_[i] = fill;
        size_ = n;
    }
    void assign(size_t n, T fill) {
        reserve(n);
        size_ = n;
        std::fill(data_, data_ + n, fill);
    }
    void push_back(T v) {
        if (size_ == cap_) reserve(cap_ * 2);
        data_[size_++] = v;
    }
    void append(const Arr& o) {
        reserve(size_ + o.size_);
        std::memcpy(data_ + size_, o.data_, o.size_ * sizeof(T));
        size_ += o.size_;
    }
    size_t size() const { return size_; }
    T* data() { return data_; }
    const T* data() const { return data_; }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }

  private:
    void release() {                                    // a heap block's capacity is a whole size class
        if (data_ != small_) arr_pool::put(data_, arr_pool::size_class(cap_ * sizeof(T)));
    }
    void take(Arr& o) {                                 // steal a heap buffer, copy inline contents
        if (o.data_ != o.small_) {
            data_ = o.data_, cap_ = o.cap_;
            o.data_ = o.small_, o.cap_ = Inline;
        } else {
            std::memcpy(small_, o.small_, o.size_ * sizeof(T));
        }
        size_ = o.size_, o.size_ = 0;
    }
    void copy(const Arr& o) {
        size_ = 0;
        reserve(o.size_);
        std::memcpy(data_, o.data_, o.size_ * sizeof(T));
        size_ = o.size_;
    }
};

}  // namespace lanesim
