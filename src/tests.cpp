#include "tests.hpp"
#include <cassert>

void test_BVH4Node_isLeaf() {
    LBVH4Node node{};

    node.children[0] = 0b1 << 31;
    node.children[1] = 0b0 << 31;

    assert(node.isLeaf(0) == true);
    assert(node.isLeaf(1) == false);
}

void test_BVH4Node_isInnerNode() {
    LBVH4Node node{};

    node.children[0] = 0b1 << 31;
    node.children[1] = 0b0 << 31;

    assert(node.isInner(0) == false);
    assert(node.isInner(1) == true);
}

void test_BVH4Node_getNumPrimitives() {
    constexpr int primitives = 16 / 4;

    LBVH4Node node{};

    node.children[0] = primitives << 27;

    assert(node.getNumPrimitives(0) == primitives * 4);
}

void test_BVH4Node_getPrimitiveIndices() {
    constexpr int index = 1493;

    LBVH4Node node{};

    node.children[0] = index;

    assert(node.getPrimitiveIndices(0) == index);
}

const TestFnPtr TEST_FN_PTRS[] = {
    test_BVH4Node_isLeaf,
    test_BVH4Node_isInnerNode,
    test_BVH4Node_getNumPrimitives,
    test_BVH4Node_getPrimitiveIndices,
};

const std::size_t TEST_FN_PTRS_SIZE = sizeof(TEST_FN_PTRS) / sizeof(TestFnPtr);