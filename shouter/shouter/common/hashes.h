#ifndef SHOUTER_HASHES_H
#define SHOUTER_HASHES_H

#include <functional>

#include "common/common.h"

// Golden ratio from http://burtleburtle.net/bob/hash/doobs.html
#define GOLDEN_RATIO 0x9e3779b9

namespace std {

namespace {

template <typename T>
inline std::size_t hash_one(const T& element, std::size_t seed) {
  return seed ^
         (std::hash<T>()(element) + GOLDEN_RATIO + (seed << 6) + (seed >> 2));
}

} // namespace

template <typename T> struct hash<std::vector<T>> {
  typedef std::vector<T> argument_type;
  typedef std::size_t result_type;

  result_type operator()(argument_type const& in) const {
    size_t size = in.size();
    result_type seed = 0;
    for (size_t i = 0; i < size; i++)
      seed = hash_one<T>(in[i], seed);
    return seed;
  }
};

template <typename U, typename V> struct hash<std::tuple<U, V>> {
  typedef std::tuple<U, V> argument_type;
  typedef std::size_t result_type;

  result_type operator()(argument_type const& in) const {
    result_type seed = 0;
    seed = hash_one<U>(std::get<0>(in), seed);
    seed = hash_one<V>(std::get<1>(in), seed);
    return seed;
  }
};

template <> struct hash<shouter::common::Framework> {
  std::size_t operator()(shouter::common::Framework const& in) const {
    return (std::size_t)in;
  }
};

} // namespace std

#endif // SHOUTER_HASHES_H
