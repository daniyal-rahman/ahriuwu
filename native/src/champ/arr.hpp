// Arr<T>: a small vector with inline storage for short arrays (the (C,) per-champion fields), heap beyond.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace lanesim {

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
    ~Arr() {
        if (data_ != small_) delete[] data_;
    }
    void reserve(size_t n) {
        if (n <= cap_) return;
        T* d = new T[n];
        std::memcpy(d, data_, size_ * sizeof(T));
        if (data_ != small_) delete[] data_;
        data_ = d, cap_ = n;
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
    void copy(const Arr& o) {
        size_ = 0;
        reserve(o.size_);
        std::memcpy(data_, o.data_, o.size_ * sizeof(T));
        size_ = o.size_;
    }
};

}  // namespace lanesim
