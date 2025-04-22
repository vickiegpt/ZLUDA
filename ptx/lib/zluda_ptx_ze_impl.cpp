// Intel Level Zero implementation of PTX functions
// Build with: icpx -fsycl -fsycl-targets=spir64 -o zluda_ptx_ze_impl.bc \
//            --offload-device-only -c zluda_ptx_ze_impl.cpp

#include <algorithm> // For std::min
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

using namespace sycl;

// Macro for exported PTX helper functions
#define FUNC(TYPE, NAME) \
  extern "C" SYCL_EXTERNAL TYPE __zluda_ptx_impl_##NAME

// Saturating arithmetic helpers
SYCL_EXTERNAL uint32_t add_sat(uint32_t x, uint32_t y) {
  uint64_t r = uint64_t(x) + uint64_t(y);
  return r > UINT32_MAX ? UINT32_MAX : uint32_t(r);
}

SYCL_EXTERNAL uint32_t sub_sat(uint32_t x, uint32_t y) {
  return x > y ? x - y : 0;
}

// PTX functions that query lane/subgroup/block properties
FUNC(uint32_t, activemask)(nd_item<3> it) {
  auto sg = it.get_sub_group();
  uint32_t sz = sg.get_local_range()[0];
  return (1U << sz) - 1U;
}

FUNC(uint32_t, sreg_tid)(nd_item<3> it, uint8_t member) {
  return uint32_t(it.get_local_id(member));
}

FUNC(uint32_t, sreg_ntid)(nd_item<3> it, uint8_t member) {
  return uint32_t(it.get_local_range(member));
}

FUNC(uint32_t, sreg_ctaid)(nd_item<3> it, uint8_t member) {
  return uint32_t(it.get_group(member));
}

FUNC(uint32_t, sreg_nctaid)(nd_item<3> it, uint8_t member) {
  return uint32_t(it.get_group_range(member));
}

// Bit-field extract (unsigned 32)
FUNC(uint32_t, bfe_u32)(uint32_t base, uint32_t pos_32, uint32_t len_32) {
  uint32_t pos = pos_32 & 0xFFU;
  uint32_t len = len_32 & 0xFFU;
  if (pos >= 32) return 0;
  if (len >= 32) return base >> pos;
  len = std::min(len, 31U);
  return (base >> pos) & ((1U << len) - 1U);
}

// Bit-field extract (unsigned 64)
FUNC(uint64_t, bfe_u64)(uint64_t base, uint32_t pos, uint32_t len) {
  if (pos >= 64) return 0;
  if (len >= 64) return base >> pos;
  len = std::min(len, 63U);
  return (base >> pos) & ((1ULL << len) - 1ULL);
}

// Bit-field extract (signed 32)
FUNC(int32_t, bfe_s32)(int32_t base, uint32_t pos_32, uint32_t len_32) {
  uint32_t pos = pos_32 & 0xFFU;
  uint32_t len = len_32 & 0xFFU;
  if (len == 0) return 0;
  if (pos >= 32) return base >> 31;
  if (len >= 32) return base >> pos;
  len = std::min(len, 31U);
  uint32_t shift = 32 - len;
  return (base << (shift - pos)) >> shift;
}

// Bit-field extract (signed 64)
FUNC(int64_t, bfe_s64)(int64_t base, uint32_t pos, uint32_t len) {
  if (len == 0) return 0;
  if (pos >= 64) return base >> 63;
  uint64_t sum = uint64_t(pos) + uint64_t(len);
  if (sum >= 64) len = (pos > 64 ? 0U : 64U - pos);
  return (base << (64U - pos - len)) >> (64U - len);
}

// Bit-field insert (32-bit)
FUNC(uint32_t, bfi_b32)(uint32_t insert, uint32_t base,
                        uint32_t pos_32, uint32_t len_32) {
  uint32_t pos = pos_32 & 0xFFU;
  uint32_t len = len_32 & 0xFFU;
  if (pos >= 32) return base;
  uint32_t mask = (len >= 32 ? UINT32_MAX : ((1U << std::min(len,31U)) - 1U)) << pos;
  return (~mask & base) | (mask & (insert << pos));
}

// Bit-field insert (64-bit)
FUNC(uint64_t, bfi_b64)(uint64_t insert, uint64_t base,
                        uint32_t pos, uint32_t len) {
  if (pos >= 64) return base;
  uint64_t mask = (len >= 64 ? UINT64_MAX : ((1ULL << std::min(len,63U)) - 1ULL)) << pos;
  return (~mask & base) | (mask & (insert << pos));
}

// Work-group barrier
FUNC(void, bar_sync)(nd_item<3> it) {
  it.barrier(access::fence_space::global_and_local);
}

// Stub for assertion failure
FUNC(void, __assertfail)(uint64_t message, uint64_t file,
                          uint32_t line, uint64_t function,
                          uint64_t char_size) {
  (void)message; (void)file; (void)line; (void)function; (void)char_size;
#ifdef __SYCL_DEVICE_ONLY__
  asm volatile("// Assertion failed\n");
  if (message == ~message) {
    volatile int x = 0; x = x / x;
  }
#endif
}
