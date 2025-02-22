#include "bvh4.hpp"

inline int encodeBVH4Leaf(const BVH2Node *leaf) {
    return -((leaf->numPrimitives << 27) | (leaf->firstPrimOffset & BVH4_INDICES_MASK));
}

void BVH4::build() {
    primitives.resize(scene.numPrimitives());

    std::vector<Primitive> bvhPrimitives(primitives.size());
    for (size_t i = 0; i < scene.triangles.size(); ++i) {
        primitives[i]    = Primitive{Primitive::TRIANGLE, i, scene.meshes[scene.triangles[i].meshIndex].tBounds(scene.triangles[i].index)};
        bvhPrimitives[i] = Primitive{Primitive::TRIANGLE, i, scene.meshes[scene.triangles[i].meshIndex].tBounds(scene.triangles[i].index)};
    }

    std::vector<Primitive> orderedPrimitives(primitives.size());

    int totalNodes             = 1;
    int orderedPrimitiveOffset = 0;

    const BVH2Node *root = buildBVH2Tree(bvhPrimitives, &totalNodes, &orderedPrimitiveOffset, orderedPrimitives, BVH4_MAX_PRIMS_IN_NODE);
    primitives.swap(orderedPrimitives);

    bvhPrimitives.resize(0);
    bvhPrimitives.shrink_to_fit();

    nodes      = new LBVH4Node[totalNodes];
    int offset = 0;
    flattenBVH2toLBVH4(root, nodes, &offset);

    root->destroy();
    delete root;
}

int flattenBVH2toLBVH4(const BVH2Node *node, LBVH4Node *nodes, int *offset) {
    const int nodeOffset  = (*offset)++;
    LBVH4Node *linearNode = &nodes[nodeOffset];

    // Check if this node is a leaf
    // In this case, we encode it as so in the parent node
    if (node->numPrimitives > 0) {
        return encodeBVH4Leaf(node);
    }

    // Otherwise, we have an inner node, in which case we need to collapse two levels
    const BVH2Node *left  = node->children[0];
    const BVH2Node *right = node->children[1];

    const BVH2Node *n[4];

    // If a child is a leaf, we use the parent as the node to be encoded (as the left child)
    n[0] = left->isLeaf() ? left : left->children[0];
    n[1] = left->isLeaf() ? nullptr : left->children[1];
    n[2] = right->isLeaf() ? right : right->children[0];
    n[3] = right->isLeaf() ? nullptr : right->children[1];

    for (size_t i = 0; i < 4; ++i) {
        if (n[i] != nullptr) {
            // Load bbox in SoA format
            const auto bbox             = n[i]->bbox;
            linearNode->bbox.pmax[0][i] = bbox.pmax.x;
            linearNode->bbox.pmax[1][i] = bbox.pmax.y;
            linearNode->bbox.pmax[2][i] = bbox.pmax.z;
            linearNode->bbox.pmax[0][i] = bbox.pmin.x;
            linearNode->bbox.pmax[1][i] = bbox.pmin.y;
            linearNode->bbox.pmax[2][i] = bbox.pmin.z;

            // Encode leaf or recurse
            if (n[i]->isLeaf()) {
                linearNode->children[i] = encodeBVH4Leaf(n[i]);
            } else {
                linearNode->children[i] = flattenBVH2toLBVH4(n[i], nodes, offset);
            }
        } else {
            // This is a leaf node, and has already been encoded on the left
            linearNode->children[i] = BVH4_INT_MIN;
        }
    }

    // Update split axes
    linearNode->axis[0] = node->splitAxis;
    linearNode->axis[1] = (left->isLeaf() ? -1 : left->splitAxis);
    linearNode->axis[2] = (right->isLeaf() ? -1 : right->splitAxis);

    return nodeOffset;
}

bool BVH4::closestHit(const Ray &r, const Interval t, SurfaceIntersection &record) const {
    float4 tMin = simd::broadcast(t.min);
    float4 tMax = simd::broadcast(t.max);

    float4 origin[3];
    bool dirIsNeg[3];
    Vec3f invDir;
    float4 invDir_4[3];

    for (auto i = 0; i < 3; ++i) {
        origin[i]   = simd::broadcast(r.origin[i]);
        invDir[i]   = 1 / r.dir[i];
        invDir_4[i] = simd::broadcast(invDir[i]);
        dirIsNeg[i] = invDir[i] < 0;
    }

    int toVisitOffset    = 0;
    int currentNodeIndex = 0;
    int stack[64];
    bool hitAnything = false;

    while (true) {
        const LBVH4Node *node = &nodes[currentNodeIndex];
        const auto [pmin, pmax]       = node->bbox;

        for (auto i = 0; i < 3; ++i) {
            tMin = simd::max(simd::mul(simd::sub(invDir[i] > 0.0f ? simd::load(pmin[i]) : simd::load(pmax[i]), origin[i]), invDir_4[i]), tMin);
            tMax = simd::min(simd::mul(simd::sub(invDir[i] > 0.0f ? simd::load(pmax[i]) : simd::load(pmin[i]), origin[i]), invDir_4[i]), tMax);
        }

        const auto mask = simd::leq(tMin, tMax);
        break;
    }

    return false;
}