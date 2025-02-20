#pragma once

#include "aabb.hpp"
#include "common.hpp"
#include "mesh.hpp"
#include "primitives.hpp"
#include "scene.hpp"


struct alignas(32) LinearBVH2Node {
    AABB bbox;
    union {
        int primitivesOffset;
        int secondChildOffset;
    };
    uint16_t numPrimitives;
    uint8_t axis;
};

struct BVH2Node {
    AABB bbox;
    BVH2Node *children[2];
    int splitAxis;
    int firstPrimOffset;
    int numPrimitives;

    void initLeaf(const int first, const int n, const AABB &bounds) {
        firstPrimOffset = first;
        numPrimitives = n;
        bbox = bounds;
        children[0] = children[1] = nullptr;
    }

    void initBranch(const int axis, BVH2Node *child0, BVH2Node *child1) {
        children[0] = child0;
        children[1] = child1;
        bbox = AABB(child0->bbox, child1->bbox);
        splitAxis = axis;
        numPrimitives = 0;
    }

    [[nodiscard]] bool isLeaf() const {
        return children[0] == nullptr && children[1] == nullptr;
    }

    bool isBranch() const {
        return !isLeaf();
    }

    void destroy() const {
        if (isBranch()) {
            children[0]->destroy();
            children[1]->destroy();

            delete children[0];
            delete children[1];
        }
    }
};

struct BVH2 {
    int maxPrimsInNode = 0;
    std::vector<Primitive> primitives;
    LinearBVH2Node *nodes = nullptr;
    const Scene &scene;

    void build();
    bool closestHit(const Ray &r, Interval t, SurfaceIntersection &record) const;
    bool anyHit(const Ray &r, Interval t) const;

    void destroy() const {
        if (nodes) delete[] nodes;
    }
};

BVH2Node *buildBVH2Tree(std::span<Primitive> bvhPrimitives, int *totalNodes, int *orderedPrimitiveOffset, std::vector<Primitive> &orderedPrimitives, int maxPrimsInNode);

int flattenBVH2(const BVH2Node *node, LinearBVH2Node *nodes, int *offset);