/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GE_COMMON_TUPLE_H_
#define GE_COMMON_TUPLE_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"

namespace ge {
template <typename ValueType>
class Tuple {
 public:
  Tuple() = default;
  inline ~Tuple() {
    delete[] data_heap_;
    data_heap_ = nullptr;
  }
  ///
  /// @brief copy constructor from another tuple
  /// @param s the source tuple
  ///
  inline Tuple(const Tuple<ValueType> &s) { this->assign(s.begin(), s.end()); }
  ///
  /// @brief constructor from initializer list
  /// @param init the initializer_list
  ///
  inline Tuple(const std::initializer_list<ValueType> &init) { this->assign(init.begin(), init.end()); }
  ///
  /// @brief constructor from vector
  /// @param init the vector
  ///
  inline Tuple(const std::vector<ValueType> &init) {  // NOLINT(runtime/explicit)
    this->assign(init.begin(), init.end());
  }
  ///
  /// @brief move constructor from Tuple
  /// @param src the source shape
  ///
  inline Tuple(Tuple<ValueType> &&src) {  // NOLINT(runtime/explicit)
    this->swap(src);
  }
  ///
  /// @brief construct the Tuple from content of iterator
  /// @param begin the beginning of iterator
  /// @param end end the end of the iterator
  /// @tparam RandomAccessIterator iterator type
  ///
  template <typename RandomAccessIterator>
  inline Tuple(RandomAccessIterator begin, RandomAccessIterator end) {
    this->assign(begin, end);
  }
  ///
  /// @brief Assign content to tuple from iterator.
  /// @param begin the beginning of iterator
  /// @param end end the end of the iterator
  /// @tparam RandomAccessIterator iterator type
  ///
  template <typename RandomAccessIterator>
  inline void assign(const RandomAccessIterator &begin, const RandomAccessIterator &end) {
    this->SetDim(end - begin);
    (void)std::copy(begin, end, this->begin());
  }
  ///
  /// @brief Swap current object with other
  /// @param other another object to be swapped.
  ///
  inline void swap(Tuple<ValueType> &other) {  // NOLINT(*)
    std::swap(ndim_, other.ndim_);
    std::swap(num_heap_allocated_, other.num_heap_allocated_);
    std::swap(data_stack_, other.data_stack_);
    std::swap(data_heap_, other.data_heap_);
  }
  ///
  /// @brief assignment from another tuple.
  /// @param src source tuple
  /// @return reference of self
  ///
  inline Tuple<ValueType> &operator=(const Tuple<ValueType> &src) {
    if (&src != this) {
      this->assign(src.begin(), src.end());
    }
    return *this;
  }
  ///
  /// @brief assignment from rvalue of another tuple.
  /// @param src source tuple
  /// @return reference of self
  ///
  inline Tuple<ValueType> &operator=(Tuple<ValueType> &&src) {
    if (&src != this) {
      Tuple<ValueType>(std::move(src)).swap(*this);
    }
    return *this;
  }
  ///
  /// @brief assignment from initializer list
  /// @param init the source initializer list
  /// @return reference of self
  ///
  inline Tuple<ValueType> &operator=(std::initializer_list<ValueType> init) {
    this->assign(init.begin(), init.end());
    return *this;
  }
  ///
  /// @return whether two tuple equals
  /// @param s the tuple to compare against
  ///
  inline bool operator==(const Tuple<ValueType> &s) const {
    if (ndim_ != s.ndim_) return false;
    return std::equal(begin(), end(), s.begin());
  }
  ///
  /// @return whether two tuple not equal
  /// @param s the tuple to compare against
  ///
  inline bool operator!=(const Tuple<ValueType> &s) const { return !(*this == s); }
  ///
  /// @return the begin data pointer to content of the tuple
  ///
  inline const ValueType *begin() const { return ndim_ <= STACK_CACHE_NUM ? data_stack_ : data_heap_; }
  ///
  /// @return the begin data pointer to content of the tuple
  ///
  inline ValueType *begin() { return ndim_ <= STACK_CACHE_NUM ? data_stack_ : data_heap_; }
  ///
  /// @return the data pointer to end of the tuple
  ///
  inline const ValueType *end() const {
    return ndim_ <= STACK_CACHE_NUM ? (data_stack_ + ndim_) : (data_heap_ + ndim_);
  }
  ///
  /// @return the data pointer to end the tuple
  ///
  inline ValueType *end() { return ndim_ <= STACK_CACHE_NUM ? (data_stack_ + ndim_) : (data_heap_ + ndim_); }
  ///
  /// @return number of dimension of the tuple
  ///
  inline uint32_t ndim() const { return ndim_; }
  ///
  /// @brief get corresponding index
  /// @param i dimension index
  /// @return the corresponding dimension size
  ///
  inline ValueType &operator[](size_t i) { return begin()[i]; }
  ///
  /// @brief get corresponding index
  /// @param i dimension index
  /// @return the corresponding dimension size
  ///
  inline const ValueType &operator[](size_t i) const { return begin()[i]; }
  ///
  /// @brief allow output string of tuple to ostream
  /// @param os the output stream
  /// @param t the tuple
  /// @return the ostream
  ///
  friend std::ostream &operator<<(std::ostream &os, const Tuple<ValueType> &t) {
    os << '[';
    const ValueType *begin = t.begin();
    const ValueType *end = t.end();
    for (const ValueType *it = begin; it != end; ++it) {
      if (it != begin) os << ',';
      os << *it;
    }
    os << ']';
    return os;
  }
  ///
  /// @brief read tuple from the istream
  /// @param is the input stream
  /// @param t The tuple
  /// @return the istream
  ///
  friend std::istream &operator>>(std::istream &is, Tuple<ValueType> &t) {
    // get (
    if (!HandleLeftBracket(is, t)) {
      return is;
    }

    // Handle empty tuple
    while (isspace(is.peek())) {
      (void)is.get();
    }
    if (IsRightBracket(is.peek())) {
      (void)is.get();
      return is;
    }
    // Handle non-empty tuple
    ValueType idx;
    std::vector<ValueType> tmp;
    while (is >> idx) {
      tmp.push_back(idx);
      char ch;
      do {
        ch = static_cast<char>(is.get());
      } while (isspace(ch));
      if (std::is_integral<ValueType>::value && ch == 'L') {
        ch = static_cast<char>(is.get());
      }
      if (ch == ',') {
        while (true) {
          ch = static_cast<char>(is.peek());
          if (isspace(ch)) {
            (void)is.get();
            continue;
          }
          if (IsRightBracket(ch)) {
            (void)is.get();
            break;
          }
          break;
        }
        if (IsRightBracket(ch)) break;
      } else if (IsRightBracket(ch)) {
        break;
      } else {
        is.setstate(std::ios::failbit);
        return is;
      }
    }
    t.assign(tmp.begin(), tmp.end());
    return is;
  }

  // stack cache size
  static const uint32_t STACK_CACHE_NUM = 4;
  // in stack space used to store shape when it is small
  ValueType data_stack_[STACK_CACHE_NUM];
  // space to store shape when dimension is big
  ValueType *data_heap_{nullptr};
  uint32_t ndim_{0};

 protected:
  // number of cells allocated in data_heap_
  uint32_t num_heap_allocated_{0};

  // internal function to change the dimension
  inline void SetDim(uint32_t ndim) {
    if (ndim > STACK_CACHE_NUM && ndim > num_heap_allocated_) {
      if (data_heap_ != nullptr) {
        delete[] data_heap_;
        data_heap_ = nullptr;
      }
      data_heap_ = new (std::nothrow) ValueType[ndim]();
      if (data_heap_ == nullptr) {
        GELOGW("data_heap_ is nullptr.");
      }
      num_heap_allocated_ = ndim;
    }
    ndim_ = ndim;
  }
  static inline bool IsLeftBracket(char ch) { return ch == '(' || ch == '['; }

  static inline bool IsRightBracket(char ch) { return ch == ')' || ch == ']'; }

  friend bool HandleLeftBracket(std::istream &is, Tuple<ValueType> &t) {
    while (true) {
      char ch = is.peek();
      if (isdigit(ch) || (ch == '-')) {
        ValueType idx;
        if (is >> idx) {
          t.assign(&idx, &idx + 1);
        }
        return false;
      }
      (void)is.get();
      if (IsLeftBracket(ch)) {
        break;
      }

      if (!isspace(ch)) {
        is.setstate(std::ios::failbit);
        return false;
      }
    }

    return true;
  }
};

using UintTuple = Tuple<uint32_t>;
using IntTuple = Tuple<int64_t>;
using FloatTuple = Tuple<float>;
using BoolTuple = Tuple<bool>;
using StringTuple = Tuple<std::string>;
}  // namespace ge

#endif  // GE_COMMON_TUPLE_H_
