/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_VEC_SWIZZLE_HPP
#define HIPSYCL_VEC_SWIZZLE_HPP

#include <type_traits>

#include "vec.hpp"
#include "vec_common.hpp"

namespace hipsycl {
namespace sycl {

template<class T, int N>
class vec;

namespace detail {

template<int Target_index, int Current_index, int...Swizzle_indices>
struct swizzle_index_accessor
{
  static constexpr int value = swizzle_index_accessor<Target_index - 1, Swizzle_indices...>::value;
};

template<int Current_index, int... Swizzle_indices>
struct swizzle_index_accessor<0, Current_index, Swizzle_indices...>
{
  static constexpr int value = Current_index;
};

template<class dataT,
         int Original_vec_size,
         int... Access_indices>
struct vec_swizzled_access
{

  static constexpr int swizzled_vec_size = sizeof...(Access_indices);
  using linear_indices_container =
    typename vector_impl<dataT, swizzled_vec_size>::indices;

  HIPSYCL_UNIVERSAL_TARGET
  vec_swizzled_access(vector_impl<dataT, Original_vec_size>& vector_ref)
    : _data{vector_ref}
  {}

  /// Sets the original vector to the given vector by applying the swizzle
  HIPSYCL_UNIVERSAL_TARGET
  void set(const vector_impl<dataT, swizzled_vec_size>& data)
  { set(linear_indices_container{}, data); }

  /// Get a swizzled vector from the original vector
  HIPSYCL_UNIVERSAL_TARGET
  vector_impl<dataT, swizzled_vec_size>
  get() const
  { return get(linear_indices_container{}); }

  /// \return The vector object that this swizzle refers to
  HIPSYCL_UNIVERSAL_TARGET
  vector_impl<dataT, Original_vec_size>& get_vector() const
  { return _data; }

  template<int Component>
  HIPSYCL_UNIVERSAL_TARGET
  static constexpr int get_swizzle_index()
  { return swizzle_index_accessor<Component, Access_indices...>::value; }

  template<int Component>
  HIPSYCL_UNIVERSAL_TARGET
  dataT& get_component()
  { return _data.template get<get_swizzle_index<Component>()>(); }

  template<int Component>
  HIPSYCL_UNIVERSAL_TARGET
  dataT get_component() const
  { return _data.template get<get_swizzle_index<Component>()>(); }

private:
  template<int... Linear_sequence>
  HIPSYCL_UNIVERSAL_TARGET
  vector_impl<dataT, swizzled_vec_size>
  get(vector_index_sequence<Linear_sequence...>) const
  {
    vector_impl<dataT, swizzled_vec_size> result;
    auto dummy_initializer = {
      (result.template set<Linear_sequence>(
          _data.template get<Access_indices>()), 0)...
    };

    return result;
  }

  template<int... Linear_sequence>
  HIPSYCL_UNIVERSAL_TARGET
  void set(vector_index_sequence<Linear_sequence...>,
           const vector_impl<dataT, swizzled_vec_size>& new_data)
  {
    auto dummy_initializer = {
      (_data.template set<Access_indices>(
          new_data.template get<Linear_sequence>()), 0)...
    };
  }

  vector_impl<dataT, Original_vec_size>& _data;
};

template<class T,
         int Original_vec_size,
         int... Access_indices>
class vec_swizzle
{
  template<int... swizzleIndices>
  HIPSYCL_UNIVERSAL_TARGET
  auto swizzle_index_sequence(detail::vector_index_sequence<swizzleIndices...>) const
  {
    return swizzle<swizzleIndices...>();
  }


public:

  template<class, int>
  friend class ::hipsycl::sycl::vec;

  static constexpr int N = sizeof...(Access_indices);

  static_assert (N > 0, "Invalid swizzle: No swizzle access indices were given");

  using swizzled_access_type =
      vec_swizzled_access<T, Original_vec_size, Access_indices...>;

  HIPSYCL_UNIVERSAL_TARGET
  swizzled_access_type& swizzled_access() const
  { return const_cast<swizzled_access_type&>(_swizzled_access); }

  HIPSYCL_UNIVERSAL_TARGET
  explicit vec_swizzle(vector_impl<T,Original_vec_size>& vector_ref)
    : _swizzled_access{vector_ref}
  {}

  vec_swizzle() = delete;
  vec_swizzle& operator=(const vec_swizzle&) = delete;

  HIPSYCL_UNIVERSAL_TARGET
  vec_swizzle& operator=(const vec<T,N>&);

  template<int Rhs_original_vec_size,
           int... Rhs_access_indices>
  HIPSYCL_UNIVERSAL_TARGET
  vec_swizzle& operator=(vec_swizzle<T, Rhs_original_vec_size, Rhs_access_indices...>&& rhs)
  {
    _swizzled_access.set(rhs.swizzled_access().get());
    return *this;
  }

  template<int n = N,
           std::enable_if_t<n == 1>* = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  operator T() const
  { return _swizzled_access.get_vector().template get<0>(); }

  HIPSYCL_UNIVERSAL_TARGET
  static constexpr int get_count()
  { return N; }

  HIPSYCL_UNIVERSAL_TARGET
  static constexpr size_t get_size()
  { return get_count() * sizeof(T); }

  // ToDo
  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, N> convert() const;

  // ToDo
  template <typename asT>
  asT as() const;

#define HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(condition, name, id) \
  template<int n = N, \
           std::enable_if_t<(id < n) && (condition)>* = nullptr> \
  HIPSYCL_UNIVERSAL_TARGET \
  T& name() \
  { return _swizzled_access.template get_component<id>(); } \
  \
  template<int n = N, \
           std::enable_if_t<(id < n) && (condition)>* = nullptr> \
  HIPSYCL_UNIVERSAL_TARGET \
  T name() const \
  { return _swizzled_access.template get_component<id>(); }

  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 4, x, 0)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 4, y, 1)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 4, z, 2)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 4, w, 3)

  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N == 4, r, 0)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N == 4, g, 1)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N == 4, b, 2)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N == 4, a, 3)

  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, s0, 0)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, s1, 1)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, s2, 2)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, s3, 3)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, s4, 4)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, s5, 5)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, s6, 6)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, s7, 7)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, s8, 8)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, s9, 9)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, sA, 10)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, sB, 11)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, sC, 12)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, sD, 13)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, sE, 14)
  HIPSYCL_DEFINE_SWIZZLE_ACCESS_IF(N <= 16, sF, 15)

  // ToDo Think about constness
  template<int... swizzleIndices>
  HIPSYCL_UNIVERSAL_TARGET
  auto swizzle() const
  {
    return vec_swizzle<T, Original_vec_size,
        (swizzled_access_type::template get_swizzle_index<swizzleIndices>())...>{
      _swizzled_access.get_vector()
    };
  }

  template<int n = N,
           std::enable_if_t<(n > 1)>* = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  auto lo() const
  {
    return swizzle_index_sequence(
          typename detail::vector_impl<T, n>::lo_indices());
  }

  template<int n = N,
           std::enable_if_t<(n > 1)>* = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  auto hi() const
  {
    return swizzle_index_sequence(
          typename detail::vector_impl<T, n>::hi_indices());
  }

  template<int n = N,
           std::enable_if_t<(n > 1)>* = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  auto even() const
  {
    return swizzle_index_sequence(
          typename detail::vector_impl<T, n>::even_indices());
  }

  template<int n = N,
           std::enable_if_t<(n > 1)>* = nullptr>
  HIPSYCL_UNIVERSAL_TARGET
  auto odd() const
  {
    return swizzle_index_sequence(
          typename detail::vector_impl<T, n>::odd_indices());
  }

#ifdef SYCL_SIMPLE_SWIZZLES
#define HIPSYCL_DEFINE_SWIZZLE2(name, minElems, i0, i1) \
  template<int n = N, \
           std::enable_if_t<(n >= minElems && n <= 4)>* = nullptr> \
  auto name() const \
  {return swizzle<i0,i1>(); }

#define HIPSYCL_DEFINE_SWIZZLE3(name, minElems, i0, i1, i2) \
  template<int n = N, \
           std::enable_if_t<(n >= minElems && n <= 4)>* = nullptr> \
  auto name() const \
  {return swizzle<i0,i1,i2>(); }

#define HIPSYCL_DEFINE_SWIZZLE4(name, i0, i1, i2, i3) \
  template<int n = N, \
           std::enable_if_t<(n == 4)>* = nullptr> \
  auto name() const \
  {return swizzle<i0,i1,i2,i3>(); }


  HIPSYCL_DEFINE_SWIZZLE2(xx, 2, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE2(yx, 2, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE2(xy, 2, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE2(yy, 2, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE2(xz, 3, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE2(yz, 3, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE2(zy, 3, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE2(zx, 3, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE2(zz, 3, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE2(xw, 4, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE2(yw, 4, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE2(zw, 4, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE2(ww, 4, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE2(wx, 4, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE2(wy, 4, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE2(wz, 4, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE3(xxx, 3, 0, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE3(yxx, 3, 1, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE3(zxx, 3, 2, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE3(xyx, 3, 0, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE3(yyx, 3, 1, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE3(zyx, 3, 2, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE3(xzx, 3, 0, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE3(yzx, 3, 1, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE3(zzx, 3, 2, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE3(xxy, 3, 0, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE3(yxy, 3, 1, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE3(zxy, 3, 2, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE3(xyy, 3, 0, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE3(yyy, 3, 1, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE3(zyy, 3, 2, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE3(xzy, 3, 0, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE3(yzy, 3, 1, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE3(zzy, 3, 2, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE3(xxz, 3, 0, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE3(yxz, 3, 1, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE3(zxz, 3, 2, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE3(xyz, 3, 0, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE3(yyz, 3, 1, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE3(zyz, 3, 2, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE3(xzz, 3, 0, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE3(yzz, 3, 1, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE3(zzz, 3, 2, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE3(xxw, 4, 0, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE3(yxw, 4, 1, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE3(zxw, 4, 2, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE3(xyw, 4, 0, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE3(yyw, 4, 1, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE3(zyw, 4, 2, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE3(xzw, 4, 0, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE3(yzw, 4, 1, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE3(zzw, 4, 2, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE3(xwx, 4, 0, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE3(ywx, 4, 1, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE3(zwx, 4, 2, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE3(xwy, 4, 0, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE3(ywy, 4, 1, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE3(zwy, 4, 2, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE3(xwz, 4, 0, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE3(ywz, 4, 1, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE3(zwz, 4, 2, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE3(wxx, 4, 3, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE3(wyx, 4, 3, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE3(wzx, 4, 3, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE3(wxy, 4, 3, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE3(wyy, 4, 3, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE3(wzy, 4, 3, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE3(wxz, 4, 3, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE3(wyz, 4, 3, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE3(wzz, 4, 3, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE3(xww, 4, 0, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE3(yww, 4, 1, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE3(zww, 4, 2, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE3(wwx, 4, 3, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE3(wwy, 4, 3, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE3(wwz, 4, 3, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE3(www, 4, 3, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xxxx, 0, 0, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rrrr, 0, 0, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yxxx, 1, 0, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(grrr, 1, 0, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zxxx, 2, 0, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(brrr, 2, 0, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wxxx, 3, 0, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(arrr, 3, 0, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xyxx, 0, 1, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rgrr, 0, 1, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yyxx, 1, 1, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(ggrr, 1, 1, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zyxx, 2, 1, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(bgrr, 2, 1, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wyxx, 3, 1, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(agrr, 3, 1, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xzxx, 0, 2, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rbrr, 0, 2, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yzxx, 1, 2, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(gbrr, 1, 2, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zzxx, 2, 2, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(bbrr, 2, 2, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wzxx, 3, 2, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(abrr, 3, 2, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xwxx, 0, 3, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rarr, 0, 3, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(ywxx, 1, 3, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(garr, 1, 3, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zwxx, 2, 3, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(barr, 2, 3, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wwxx, 3, 3, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(aarr, 3, 3, 0, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xxyx, 0, 0, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rrgr, 0, 0, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yxyx, 1, 0, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(grgr, 1, 0, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zxyx, 2, 0, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(brgr, 2, 0, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wxyx, 3, 0, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(argr, 3, 0, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xyyx, 0, 1, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rggr, 0, 1, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yyyx, 1, 1, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(gggr, 1, 1, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zyyx, 2, 1, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(bggr, 2, 1, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wyyx, 3, 1, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(aggr, 3, 1, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xzyx, 0, 2, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rbgr, 0, 2, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yzyx, 1, 2, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(gbgr, 1, 2, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zzyx, 2, 2, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(bbgr, 2, 2, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wzyx, 3, 2, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(abgr, 3, 2, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xwyx, 0, 3, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(ragr, 0, 3, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(ywyx, 1, 3, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(gagr, 1, 3, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zwyx, 2, 3, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(bagr, 2, 3, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wwyx, 3, 3, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(aagr, 3, 3, 1, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xxzx, 0, 0, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rrbr, 0, 0, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yxzx, 1, 0, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(grbr, 1, 0, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zxzx, 2, 0, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(brbr, 2, 0, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wxzx, 3, 0, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(arbr, 3, 0, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xyzx, 0, 1, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rgbr, 0, 1, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yyzx, 1, 1, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(ggbr, 1, 1, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zyzx, 2, 1, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(bgbr, 2, 1, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wyzx, 3, 1, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(agbr, 3, 1, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xzzx, 0, 2, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rbbr, 0, 2, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yzzx, 1, 2, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(gbbr, 1, 2, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zzzx, 2, 2, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(bbbr, 2, 2, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wzzx, 3, 2, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(abbr, 3, 2, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xwzx, 0, 3, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rabr, 0, 3, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(ywzx, 1, 3, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(gabr, 1, 3, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zwzx, 2, 3, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(babr, 2, 3, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wwzx, 3, 3, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(aabr, 3, 3, 2, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xxwx, 0, 0, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rrar, 0, 0, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yxwx, 1, 0, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(grar, 1, 0, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zxwx, 2, 0, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(brar, 2, 0, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wxwx, 3, 0, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(arar, 3, 0, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xywx, 0, 1, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rgar, 0, 1, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yywx, 1, 1, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(ggar, 1, 1, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zywx, 2, 1, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(bgar, 2, 1, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wywx, 3, 1, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(agar, 3, 1, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xzwx, 0, 2, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(rbar, 0, 2, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(yzwx, 1, 2, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(gbar, 1, 2, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zzwx, 2, 2, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(bbar, 2, 2, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wzwx, 3, 2, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(abar, 3, 2, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xwwx, 0, 3, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(raar, 0, 3, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(ywwx, 1, 3, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(gaar, 1, 3, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(zwwx, 2, 3, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(baar, 2, 3, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(wwwx, 3, 3, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(aaar, 3, 3, 3, 0)
  HIPSYCL_DEFINE_SWIZZLE4(xxxy, 0, 0, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rrrg, 0, 0, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yxxy, 1, 0, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(grrg, 1, 0, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zxxy, 2, 0, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(brrg, 2, 0, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wxxy, 3, 0, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(arrg, 3, 0, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xyxy, 0, 1, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rgrg, 0, 1, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yyxy, 1, 1, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(ggrg, 1, 1, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zyxy, 2, 1, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(bgrg, 2, 1, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wyxy, 3, 1, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(agrg, 3, 1, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xzxy, 0, 2, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rbrg, 0, 2, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yzxy, 1, 2, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(gbrg, 1, 2, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zzxy, 2, 2, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(bbrg, 2, 2, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wzxy, 3, 2, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(abrg, 3, 2, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xwxy, 0, 3, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rarg, 0, 3, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(ywxy, 1, 3, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(garg, 1, 3, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zwxy, 2, 3, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(barg, 2, 3, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wwxy, 3, 3, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(aarg, 3, 3, 0, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xxyy, 0, 0, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rrgg, 0, 0, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yxyy, 1, 0, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(grgg, 1, 0, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zxyy, 2, 0, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(brgg, 2, 0, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wxyy, 3, 0, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(argg, 3, 0, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xyyy, 0, 1, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rggg, 0, 1, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yyyy, 1, 1, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(gggg, 1, 1, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zyyy, 2, 1, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(bggg, 2, 1, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wyyy, 3, 1, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(aggg, 3, 1, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xzyy, 0, 2, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rbgg, 0, 2, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yzyy, 1, 2, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(gbgg, 1, 2, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zzyy, 2, 2, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(bbgg, 2, 2, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wzyy, 3, 2, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(abgg, 3, 2, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xwyy, 0, 3, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(ragg, 0, 3, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(ywyy, 1, 3, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(gagg, 1, 3, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zwyy, 2, 3, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(bagg, 2, 3, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wwyy, 3, 3, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(aagg, 3, 3, 1, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xxzy, 0, 0, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rrbg, 0, 0, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yxzy, 1, 0, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(grbg, 1, 0, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zxzy, 2, 0, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(brbg, 2, 0, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wxzy, 3, 0, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(arbg, 3, 0, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xyzy, 0, 1, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rgbg, 0, 1, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yyzy, 1, 1, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(ggbg, 1, 1, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zyzy, 2, 1, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(bgbg, 2, 1, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wyzy, 3, 1, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(agbg, 3, 1, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xzzy, 0, 2, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rbbg, 0, 2, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yzzy, 1, 2, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(gbbg, 1, 2, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zzzy, 2, 2, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(bbbg, 2, 2, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wzzy, 3, 2, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(abbg, 3, 2, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xwzy, 0, 3, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rabg, 0, 3, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(ywzy, 1, 3, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(gabg, 1, 3, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zwzy, 2, 3, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(babg, 2, 3, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wwzy, 3, 3, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(aabg, 3, 3, 2, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xxwy, 0, 0, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rrag, 0, 0, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yxwy, 1, 0, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(grag, 1, 0, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zxwy, 2, 0, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(brag, 2, 0, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wxwy, 3, 0, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(arag, 3, 0, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xywy, 0, 1, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rgag, 0, 1, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yywy, 1, 1, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(ggag, 1, 1, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zywy, 2, 1, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(bgag, 2, 1, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wywy, 3, 1, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(agag, 3, 1, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xzwy, 0, 2, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(rbag, 0, 2, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(yzwy, 1, 2, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(gbag, 1, 2, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zzwy, 2, 2, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(bbag, 2, 2, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wzwy, 3, 2, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(abag, 3, 2, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xwwy, 0, 3, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(raag, 0, 3, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(ywwy, 1, 3, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(gaag, 1, 3, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(zwwy, 2, 3, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(baag, 2, 3, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(wwwy, 3, 3, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(aaag, 3, 3, 3, 1)
  HIPSYCL_DEFINE_SWIZZLE4(xxxz, 0, 0, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rrrb, 0, 0, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yxxz, 1, 0, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(grrb, 1, 0, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zxxz, 2, 0, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(brrb, 2, 0, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wxxz, 3, 0, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(arrb, 3, 0, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xyxz, 0, 1, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rgrb, 0, 1, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yyxz, 1, 1, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(ggrb, 1, 1, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zyxz, 2, 1, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(bgrb, 2, 1, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wyxz, 3, 1, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(agrb, 3, 1, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xzxz, 0, 2, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rbrb, 0, 2, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yzxz, 1, 2, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(gbrb, 1, 2, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zzxz, 2, 2, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(bbrb, 2, 2, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wzxz, 3, 2, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(abrb, 3, 2, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xwxz, 0, 3, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rarb, 0, 3, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(ywxz, 1, 3, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(garb, 1, 3, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zwxz, 2, 3, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(barb, 2, 3, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wwxz, 3, 3, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(aarb, 3, 3, 0, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xxyz, 0, 0, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rrgb, 0, 0, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yxyz, 1, 0, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(grgb, 1, 0, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zxyz, 2, 0, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(brgb, 2, 0, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wxyz, 3, 0, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(argb, 3, 0, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xyyz, 0, 1, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rggb, 0, 1, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yyyz, 1, 1, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(gggb, 1, 1, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zyyz, 2, 1, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(bggb, 2, 1, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wyyz, 3, 1, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(aggb, 3, 1, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xzyz, 0, 2, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rbgb, 0, 2, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yzyz, 1, 2, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(gbgb, 1, 2, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zzyz, 2, 2, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(bbgb, 2, 2, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wzyz, 3, 2, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(abgb, 3, 2, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xwyz, 0, 3, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(ragb, 0, 3, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(ywyz, 1, 3, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(gagb, 1, 3, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zwyz, 2, 3, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(bagb, 2, 3, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wwyz, 3, 3, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(aagb, 3, 3, 1, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xxzz, 0, 0, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rrbb, 0, 0, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yxzz, 1, 0, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(grbb, 1, 0, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zxzz, 2, 0, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(brbb, 2, 0, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wxzz, 3, 0, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(arbb, 3, 0, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xyzz, 0, 1, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rgbb, 0, 1, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yyzz, 1, 1, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(ggbb, 1, 1, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zyzz, 2, 1, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(bgbb, 2, 1, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wyzz, 3, 1, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(agbb, 3, 1, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xzzz, 0, 2, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rbbb, 0, 2, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yzzz, 1, 2, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(gbbb, 1, 2, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zzzz, 2, 2, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(bbbb, 2, 2, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wzzz, 3, 2, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(abbb, 3, 2, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xwzz, 0, 3, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rabb, 0, 3, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(ywzz, 1, 3, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(gabb, 1, 3, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zwzz, 2, 3, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(babb, 2, 3, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wwzz, 3, 3, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(aabb, 3, 3, 2, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xxwz, 0, 0, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rrab, 0, 0, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yxwz, 1, 0, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(grab, 1, 0, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zxwz, 2, 0, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(brab, 2, 0, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wxwz, 3, 0, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(arab, 3, 0, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xywz, 0, 1, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rgab, 0, 1, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yywz, 1, 1, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(ggab, 1, 1, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zywz, 2, 1, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(bgab, 2, 1, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wywz, 3, 1, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(agab, 3, 1, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xzwz, 0, 2, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(rbab, 0, 2, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(yzwz, 1, 2, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(gbab, 1, 2, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zzwz, 2, 2, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(bbab, 2, 2, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wzwz, 3, 2, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(abab, 3, 2, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xwwz, 0, 3, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(raab, 0, 3, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(ywwz, 1, 3, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(gaab, 1, 3, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(zwwz, 2, 3, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(baab, 2, 3, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(wwwz, 3, 3, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(aaab, 3, 3, 3, 2)
  HIPSYCL_DEFINE_SWIZZLE4(xxxw, 0, 0, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rrra, 0, 0, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yxxw, 1, 0, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(grra, 1, 0, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zxxw, 2, 0, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(brra, 2, 0, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wxxw, 3, 0, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(arra, 3, 0, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xyxw, 0, 1, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rgra, 0, 1, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yyxw, 1, 1, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(ggra, 1, 1, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zyxw, 2, 1, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(bgra, 2, 1, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wyxw, 3, 1, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(agra, 3, 1, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xzxw, 0, 2, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rbra, 0, 2, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yzxw, 1, 2, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(gbra, 1, 2, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zzxw, 2, 2, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(bbra, 2, 2, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wzxw, 3, 2, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(abra, 3, 2, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xwxw, 0, 3, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rara, 0, 3, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(ywxw, 1, 3, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(gara, 1, 3, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zwxw, 2, 3, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(bara, 2, 3, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wwxw, 3, 3, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(aara, 3, 3, 0, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xxyw, 0, 0, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rrga, 0, 0, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yxyw, 1, 0, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(grga, 1, 0, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zxyw, 2, 0, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(brga, 2, 0, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wxyw, 3, 0, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(arga, 3, 0, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xyyw, 0, 1, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rgga, 0, 1, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yyyw, 1, 1, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(ggga, 1, 1, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zyyw, 2, 1, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(bgga, 2, 1, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wyyw, 3, 1, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(agga, 3, 1, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xzyw, 0, 2, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rbga, 0, 2, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yzyw, 1, 2, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(gbga, 1, 2, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zzyw, 2, 2, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(bbga, 2, 2, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wzyw, 3, 2, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(abga, 3, 2, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xwyw, 0, 3, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(raga, 0, 3, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(ywyw, 1, 3, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(gaga, 1, 3, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zwyw, 2, 3, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(baga, 2, 3, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wwyw, 3, 3, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(aaga, 3, 3, 1, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xxzw, 0, 0, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rrba, 0, 0, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yxzw, 1, 0, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(grba, 1, 0, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zxzw, 2, 0, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(brba, 2, 0, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wxzw, 3, 0, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(arba, 3, 0, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xyzw, 0, 1, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rgba, 0, 1, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yyzw, 1, 1, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(ggba, 1, 1, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zyzw, 2, 1, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(bgba, 2, 1, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wyzw, 3, 1, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(agba, 3, 1, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xzzw, 0, 2, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rbba, 0, 2, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yzzw, 1, 2, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(gbba, 1, 2, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zzzw, 2, 2, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(bbba, 2, 2, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wzzw, 3, 2, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(abba, 3, 2, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xwzw, 0, 3, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(raba, 0, 3, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(ywzw, 1, 3, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(gaba, 1, 3, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zwzw, 2, 3, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(baba, 2, 3, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wwzw, 3, 3, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(aaba, 3, 3, 2, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xxww, 0, 0, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rraa, 0, 0, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yxww, 1, 0, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(graa, 1, 0, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zxww, 2, 0, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(braa, 2, 0, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wxww, 3, 0, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(araa, 3, 0, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xyww, 0, 1, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rgaa, 0, 1, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yyww, 1, 1, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(ggaa, 1, 1, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zyww, 2, 1, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(bgaa, 2, 1, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wyww, 3, 1, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(agaa, 3, 1, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xzww, 0, 2, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(rbaa, 0, 2, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(yzww, 1, 2, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(gbaa, 1, 2, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zzww, 2, 2, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(bbaa, 2, 2, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wzww, 3, 2, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(abaa, 3, 2, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(xwww, 0, 3, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(raaa, 0, 3, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(ywww, 1, 3, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(gaaa, 1, 3, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(zwww, 2, 3, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(baaa, 2, 3, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(wwww, 3, 3, 3, 3)
  HIPSYCL_DEFINE_SWIZZLE4(aaaa, 3, 3, 3, 3)
#endif

  // ToDo: load and store member functions
  template <access::address_space addressSpace>
  HIPSYCL_UNIVERSAL_TARGET
  void load(size_t offset, multi_ptr<T, addressSpace> ptr);

  template <access::address_space addressSpace>
  HIPSYCL_UNIVERSAL_TARGET
  void store(size_t offset, multi_ptr<T, addressSpace> ptr) const;

private:
  swizzled_access_type _swizzled_access;
};

}
}
}

#endif
