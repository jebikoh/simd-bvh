#pragma once

#ifdef USE_NEON
#include <arm_neon.h>
#elif defined(USE_SSE)
#include <smmintrin.h>
#endif


namespace simd {
#ifdef USE_NEON
using float4 = float32x4_t;
using uint4 = uint32x4_t;
#elif defined(USE_SSE)
using float4 = __m128;
#endif

/**
 * Broadcast a scalar to a float4
 * @param x scalar
 * @return vector with all lanes set to x
 */
inline float4 broadcast(const float x) {
  return vdupq_n_f32(x);
}

/**
 * Load a float4 from memory
 * @param x pointer to memory
 * @return vector loaded from memory
 */
inline float4 load(const float *x) {
  return vld1q_f32(x);
}

/**
 * Stores a float4 to memory
 * @param p pointer to memory
 * @param v vector to store
 */
inline void store(float32_t *p, const float4 v) {
  vst1q_f32(p, v);
}

/**
 * Vector addition
 * @param a LHS
 * @param b RHS
 * @return vector sum
 */
inline float4 add(const float4 a, const float4 b) {
  return vaddq_f32(a, b);
}

/**
 * Vector subtraction
 * @param a LHS
 * @param b RHS
 * @return vector difference
 */
inline float4 sub(const float4 a, const float4 b) {
  return vsubq_f32(a, b);
}

/**
 * Vector multiplication
 * @param a LHS
 * @param b RHS
 * @return vector product
 */
inline float4 mul(const float4 a, const float4 b) {
  return vmulq_f32(a, b);
}

/**
 * Vector division
 * @param a LHS
 * @param b RHS
 * @return vector quotient
 */
inline float4 div(const float4 a, const float4 b) {
  return vdivq_f32(a, b);
}

/**
 * Extended vector multiplication
 * @param a LHS
 * @param b RHS
 * @return vector product
 */
inline float4 mulExt(const float4 a, const float4 b) {
  return vmulxq_f32(a, b);
}

/**
 * Computes multiply-add to accumulator: a + (b * c)
 * @param a vector
 * @param b vector
 * @param c vector
 * @return a + (b * c)
 */
inline float4 mulAddAcc(const float4 a, const float4 b, const float4 c) {
  return vmlaq_f32(a, b, c);
}

/**
 * Computes multiply-subtract to accumulator: a - (b * c)
 * @param a vector
 * @param b vector
 * @param c vector
 * @return a - (b * c)
 */
inline float4 mulSubAcc(const float4 a, const float4 b, const float4 c) {
  return vmlsq_f32(a, b, c);
}

/**
 * Computes fused multiply-add: a * b + c
 * @param a vector
 * @param b vector
 * @param c vector
 * @return FMA result
 */
inline float4 fma(const float4 a, const float4 b, const float4 c) {
  return vfmaq_f32(c, a, b);
}

/**
 * Computes fused multiply-subtract: a * b - c
 * @param a vector
 * @param b vector
 * @param c vector
 * @return FMS result
 */
inline float4 fms(const float4 a, const float4 b, const float4 c) {
  return vfmsq_f32(c, a, b);
}

/**
 * Computes absolute value of the difference of two vectors
 * @param a LHS
 * @param b RHS
 * @return absolute difference
 */
inline float4 absDiff(const float4 a, const float4 b) {
  return vabdq_f32(a, b);
}

/**
 * Computes absolute value of a vector
 * @param a vector
 * @return absolute value of the vector
 */
inline float4 abs(const float4 a) {
  return vabsq_f32(a);
}

/**
 * Computes the maximum of two vectors
 * @param a vector
 * @param b vector
 * @return vector of max(a, b)
 */
inline float4 max(const float4 a, const float4 b) {
  return vmaxq_f32(a, b);
}

/**
 * Computes the minimum of two vectors
 * @param a vector
 * @param b vector
 * @return vector of min(a, b)
 */
inline float4 min(const float4 a, const float4 b) {
  return vminq_f32(a, b);
}

/**
 * Computes the maximum of two vectors, adhering to IEEE 754 standard
 * @param a vector
 * @param b vector
 * @return vector of max(a, b)
 */
inline float4 maxNm(const float4 a, const float4 b) {
  return vmaxnmq_f32(a, b);
}

/**
 * Computes the minimum of two vectors, adhering to IEEE 754 standard
 * @param a vector
 * @param b vector
 * @return vector of min(a, b)
 */
inline float4 minNm(const float4 a, const float4 b) {
  return vminnmq_f32(a, b);
}

/**
 * Truncates the floating point values to integers (towards zero)
 * @param a
 * @return
 */
inline float4 truncate(const float4 a) {
  return vrndq_f32(a);
}

/**
 * Floating point round to nearest integer, with ties to even
 * @param a
 * @return
 */
inline float4 round(const float4 a) {
  return vrndnq_f32(a);
}

/**
 * Floors the floating point values to integers (neg infinity)
 * @param a
 * @return
 */
inline float4 floor(const float4 a) {
  return vrndmq_f32(a);
}

/**
 * Ceils the floating point values to integers (pos infinity)
 * @param a
 * @return
 */
inline float4 ceil(const float4 a) {
  return vrndpq_f32(a);
}

/**
 * Computes the reciprocal estimate
 * @param a vector
 * @return reciprocal estimate of the vector
 */
inline float4 reciprocal(const float4 a) {
  return vrecpeq_f32(a);
}

/**
 * Computes FP reciprocal step: 2.0 - a * b
 * @param a vector
 * @param b vector
 * @return reciprocal step
 */
inline float4 reciprocal(const float4 a, const float4 b) {
  return vrecpsq_f32(a, b);
}

/**
 * Computes the reciprocal square root estimate
 * @param a vector
 * @return reciprocal square root estimate of the vector
 */
inline float4 reciprocalSqrt(const float4 a) {
  return vrsqrteq_f32(a);
}

/**
 * Computes the FP reciprocal square root step: (3.0 - a * b) / 2.0
 * @param a vector
 * @param b vector
 * @return reciprocal square root step
 */
inline float4 reciprocalSqrt(const float4 a, const float4 b) {
  return vrsqrtsq_f32(a, b);
}

/**
 * Computes the square root
 * @param a vector
 * @return square root of the vector
 */
inline float4 sqrt(const float4 a) {
  return vsqrtq_f32(a);
}

/**
 * Performs pairwise addition of two vectors.
 *
 * For vectors a = [a0, a1, a2, a3] and b = [b0, b1, b2, b3]
 * the result is [a0 + a1, a2 + a3, b0 + b1, b2 + b3]
 *
 * @param a first vector
 * @param b second vector
 * @return pairwise sum of the vectors
 */
inline float4 pairwiseAdd(const float4 a, const float4 b) {
  return vpaddq_f32(a, b);
}

/**
 * Performs pairwise max of two vectors
 *
 * For vectors a = [a0, a1, a2, a3] and b = [b0, b1, b2, b3]
 * the result is [max(a0, a1), max(a2, a3), max(b0, b1), max(b2, b3)]
 *
 * @param a first vector
 * @param b second vector
 * @return pairwise max of the vectors
 */
inline float4 pairwiseMax(const float4 a, const float4 b) {
  return vpmaxq_f32(a, b);
}

/**
 * Performs pairwise min of two vectors
 *
 * For vectors a = [a0, a1, a2, a3] and b = [b0, b1, b2, b3]
 * the result is [min(a0, a1), min(a2, a3), min(b0, b1), min(b2, b3)]
 *
 * @param a first vector
 * @param b second vector
 * @return pairwise min of the vectors
 */
inline float4 pairwiseMin(const float4 a, const float4 b) {
  return vpminq_f32(a, b);
}

/**
 * Performs pairwise max of two vectors, adhering to IEEE 754 standard
 *
 * For vectors a = [a0, a1, a2, a3] and b = [b0, b1, b2, b3]
 * the result is [max(a0, a1), max(a2, a3), max(b0, b1), max(b2, b3)]
 *
 * @param a first vector
 * @param b second vector
 * @return pairwise max of the vectors
 */
inline float4 pairwiseMaxStrict(const float4 a, const float4 b) {
  return vpmaxnmq_f32(a, b);
}

/**
 * Performs pairwise min of two vectors, adhering to IEEE 754 standard
 *
 * For vectors a = [a0, a1, a2, a3] and b = [b0, b1, b2, b3]
 * the result is [min(a0, a1), min(a2, a3), min(b0, b1), min(b2, b3)]
 *
 * @param a first vector
 * @param b second vector
 * @return pairwise min of the vectors
 */
inline float4 pairwiseMinStrict(const float4 a, const float4 b) {
  return vpminnmq_f32(a, b);
}

/**
 * Sums the elements of a vector
 * @param a vector
 * @return horizontal sum of the vector
 */
inline float32_t sum(const float4 a) {
  return vaddvq_f32(a);
}

/**
 * Computes the maximum element of a vector
 * @param a vector
 * @return maximum element of the vector
 */
inline float32_t max(const float4 a) {
  return vmaxvq_f32(a);
}

/**
 * Computes the minimum element of a vector
 * @param a vector
 * @return minimum element of the vector
 */
inline float32_t min(const float4 a) {
  return vmaxvq_f32(a);
}

/**
 * Computes the maximum element of a vector, adhering to IEEE 754 standard
 * @param a vector
 * @return maximum element of the vector
 */
inline float32_t maxStrict(const float4 a) {
  return vmaxnmvq_f32(a);
}

/**
 * Computes the minimum element of a vector, adhering to IEEE 754 standard
 * @param a vector
 * @return minimum element of the vector
 */
inline float32_t minStrict(const float4 a) {
  return vminnmvq_f32(a);
}

/**
 * Checks if two vectors are equal (bitwise)
 * @param a LHS
 * @param b RHS
 * @return vector of equalities per pair
 */
inline uint4 equal(const float4 a, const float4 b) {
  return vceqq_f32(a, b);
}

/**
 * Checks if a vector is equal to zero
 * @param a vector
 * @return vector of equalities to zero
 */
inline uint32x4_t equalZero(const float4 a) {
  return vceqzq_f32(a);
}

/**
 * Checks if LHS vector is >= RHS vector
 * @param a LHS
 * @param b RHS
 * @return vector of comparisons per pair
 */
inline uint4 geq (const float4 a, const float4 b) {
  return vcgeq_f32(a, b);
}

/**
 * Checks if a vector is >=0
 * @param a vector
 * @return vector of comparisons to zero
 */
inline uint4 geqZero(const float4 a) {
  return vcgezq_f32(a);
}

/**
 * Checks if LHS vector is <= RHS vector
 * @param a LHS
 * @param b RHS
 * @return vector of comparisons per pair
 */
inline uint4 leq(const float4 a, const float4 b) {
  return vcleq_f32(a, b);
}

/**
 * Checks if a vector is <=0
 * @param a vector
 * @return vector of comparisons to zero
 */
inline uint4 leqZero(const float4 a) {
  return vclezq_f32(a);
}

/**
 * Checks if LHS vector is > RHS vector
 * @param a LHS
 * @param b RHS
 * @return vector of comparisons per pair
 */
inline uint4 gt(const float4 a, const float4 b) {
  return vcgtq_f32(a, b);
}

/**
 * Checks if a vector is >0
 * @param a vector
 * @return vector of comparisons to zero
 */
inline uint4 gtZero(const float4 a) {
  return vcgtzq_f32(a);
}

/**
 * Checks if RHS vector is < LHS vector
 * @param a LHS
 * @param b RHS
 * @return vector of comparisons per pair
 */
inline uint4 lt(const float4 a, const float4 b) {
  return vcltq_f32(a, b);
}

/**
 * Checks if a vector is <0
 * @param a vector
 * @return vector of comparisons to zero
 */
inline uint4 ltZero(const float4 a) {
  return vcltzq_f32(a);
}

/**
 * Checks if the vector abs(a) >= abs(b)
 * @param a LHS
 * @param b RHS
 * @return vector of comparisons per pair
 */
inline uint4 absGeq(const float4 a, const float4 b) {
  return vcageq_f32(a, b);
}

/**
 * Checks if the vector abs(a) <= abs(b)
 * @param a LHS
 * @param b RHS
 * @return vector of comparisons per pair
 */
inline uint4 absLeq(const float4 a, const float4 b) {
  return vcaleq_f32(a, b);
}

/**
 * Checks if the vector abs(a) > abs(b)
 * @param a LHS
 * @param b RHS
 * @return vector of comparisons per pair
 */
inline uint4 absGt(const float4 a, const float4 b) {
  return vcagtq_f32(a, b);
}

/**
 * Checks if the vector abs(a) < abs(b)
 * @param a LHS
 * @param b RHS
 * @return vector of comparisons per pair
 */
inline uint4 absLt(const float4 a, const float4 b) {
  return vcaltq_f32(a, b);
}

// TODO: scalar operations if needed

} // namespace simd