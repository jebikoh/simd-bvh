#pragma once

#include "simd.hpp"

// QBVH: https://www.uni-ulm.de/fileadmin/website_uni_ulm/iui.inst.100/institut/Papers/QBVH.pdf

using float4 = simd::float4;

static constexpr int BVH4_PRIMITIVE_MASK = 0xF;
static constexpr int BVH4_INDICES_MASK = 0x7FFFFFF;
static constexpr int BVH4_INT_MIN = 0x80000000;

/**
 * Holds 4 bounding boxes, stored in SoA format
 */
struct alignas(128) BVH4Node {
    // Min
    float4 xMin;
    float4 yMin;
    float4 zMin;

    // Max
    float4 xMax;
    float4 yMax;
    float4 zMax;

    /**
     * Children indices
     *  - If the index is negative, it is a leaf (<0)
     *  - If the index is 0 or positive, it is an inner node (>=0)
     */
    int children[4];

    // Split axes
    int axis[3];

    /**
     * Checks if the child at the given index is a leaf.
     * @param child child index [0, 4)
     * @return true if leaf, false otherwise
     */
    bool isLeaf(const int child) const {
        return children[child] < 0;
    }

    /**
     * Checks if the child at the given index is an inner node.
     * @param child child index [0, 4)
     * @return true if inner node, false otherwise
     */
    bool isInner(const int child) const {
        return children[child] >= 0;
    }

    /**
     * Retrieves the number of primitives in a leaf.
     * Does not check if child is actually a leaf.
     *
     * The number of children are stored in the 4-bits after the sign bit.
     * @param child child index [0, 4)
     * @return number of primitives in leaf
     */
    int getNumPrimitives(const int child) const {
        // # of primitives is always a multiple of 4 (via padding if needed)
        return ((children[child] >> 27) & BVH4_PRIMITIVE_MASK) * 4;
    }

    /**
     * Retrieves the primitive indices in a leaf.
     * Does not check if child is actually a leaf.
     *
     * The indices are stored in the lower 27 bits.
     * @param child child index [0, 4)
     * @return primitive indices
     */
    int getPrimitiveIndices(const int child) const {
        return children[child] & BVH4_INDICES_MASK;
    }
};