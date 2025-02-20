#include "bvh2.hpp"

struct BVH2Bucket {
    int count = 0;
    AABB bounds;
};


void BVH2::build() {
    primitives.resize(scene.numPrimitives());

    // BVH Primitives is our working span of primitives
    // This will start out as all of them
    std::vector<Primitive> bvhPrimitives(primitives.size());
    // for (size_t i = 0; i < scene..size(); ++i) {
    //     primitives_[i]   = Primitive{Primitive::SPHERE, i, spheres[i].bounds()};
    //     bvhPrimitives[i] = Primitive{Primitive::SPHERE, i, spheres[i].bounds()};
    // }
    // const size_t tOffset = spheres.size();
    for (size_t i = 0; i < scene.triangles.size(); ++i) {
        primitives[i]   = Primitive{Primitive::TRIANGLE, i, scene.meshes[scene.triangles[i].meshIndex].tBounds(scene.triangles[i].index)};
        bvhPrimitives[i] = Primitive{Primitive::TRIANGLE, i, scene.meshes[scene.triangles[i].meshIndex].tBounds(scene.triangles[i].index)};
    }
    // Add rest of types when we get them
    // We will order as we build
    std::vector<Primitive> orderedPrimitives(primitives.size());

    int totalNodes             = 1;
    int orderedPrimitiveOffset = 0;

    const BVH2Node *root = buildBVH2Tree(bvhPrimitives, &totalNodes, &orderedPrimitiveOffset, orderedPrimitives, maxPrimsInNode);
    primitives.swap(orderedPrimitives);

    bvhPrimitives.resize(0);
    bvhPrimitives.shrink_to_fit();

    nodes     = new LinearBVH2Node[totalNodes];
    int offset = 0;
    flattenBVH2(root, nodes, &offset);

    // Clean-up the tree
    root->destroy();
    delete root;
}

bool BVH2::closestHit(const Ray &r, Interval t, SurfaceIntersection &record) const {
    const auto invDir     = 1 / r.dir;
    const int dirIsNeg[3] = {static_cast<int>(invDir.x < 0), static_cast<int>(invDir.y < 0), static_cast<int>(invDir.z < 0)};

    int toVisitOffset    = 0;
    int currentNodeIndex = 0;
    int stack[64];
    bool hitAnything = false;

    while (true) {
        const LinearBVH2Node *node = &nodes[currentNodeIndex];
        // 1. Check the ray intersects the current node
        //    If it doesn't, pop the stack and continue
        if (node->bbox.hit(r.origin, r.dir, t)) {
            // 2. If we are at a leaf node, loop through all primitives
            //    Otherwise, push the children onto the stack
            if (node->numPrimitives > 0) {
                // Leaf node
                for (int i = 0; i < node->numPrimitives; ++i) {
                    const auto primitive = primitives[node->primitivesOffset + i];
                    bool closestHitPrim = false;

                    switch(primitive.type) {
                        case Primitive::TRIANGLE: {
                            const Triangle &triangle = scene.triangles[primitive.index];
                            float u, v;
                            closestHitPrim =  scene.meshes[triangle.meshIndex].tClosestHit(r, t, record, triangle.index, u, v);
                        }
                        default:
                            break;
                    }

                    if (closestHitPrim) {
                        hitAnything = true;
                        t.max       = record.t;
                    }
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = stack[--toVisitOffset];
            } else {
                // Interior node
                if (dirIsNeg[node->axis]) {
                    stack[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex       = node->secondChildOffset;
                } else {
                    stack[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex       = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = stack[--toVisitOffset];
        }
    }

    return hitAnything;
}

bool BVH2::anyHit(const Ray &r, Interval t) const {
    const auto invDir     = 1 / r.dir;
    const int dirIsNeg[3] = {static_cast<int>(invDir.x < 0), static_cast<int>(invDir.y < 0), static_cast<int>(invDir.z < 0)};

    int toVisitOffset    = 0;
    int currentNodeIndex = 0;
    int stack[64];

    while (true) {
        const LinearBVH2Node *node = &nodes[currentNodeIndex];
        if (node->bbox.hit(r.origin, r.dir, t)) {
            if (node->numPrimitives > 0) {
                for (int i = 0; i < node->numPrimitives; ++i) {
                    bool anyHitPrim = false;
                    switch (const auto primitive = primitives[node->primitivesOffset + i]; primitive.type) {
                        case Primitive::TRIANGLE: {
                            const Triangle &triangle = scene.triangles[primitive.index];
                            anyHitPrim = scene.meshes[triangle.meshIndex].tAnyHit(r, t, triangle.index);
                        }
                        default:
                            break;
                    }
                    if (anyHitPrim) return true;
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = stack[--toVisitOffset];
            } else {
                // Interior node
                if (dirIsNeg[node->axis]) {
                    stack[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex       = node->secondChildOffset;
                } else {
                    stack[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex       = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = stack[--toVisitOffset];
        }
    }

    return false;
}

BVH2Node *buildBVH2Tree(std::span<Primitive> bvhPrimitives, int *totalNodes, int *orderedPrimitiveOffset, std::vector<Primitive> &orderedPrimitives, int maxPrimsInNode) {
    const auto node = new BVH2Node();
    (*totalNodes)++;

    AABB bounds;
    for (const auto &prim: bvhPrimitives) {
        bounds.expand(prim.bounds);
    }

    if (bounds.surfaceArea() == 0 || bvhPrimitives.size() == 1) {
        // CASE: single prim or empty bbox;
        const int firstOffset = *orderedPrimitiveOffset;
        *orderedPrimitiveOffset += bvhPrimitives.size();
        for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
            // const int index                    = ;
            orderedPrimitives[firstOffset + i] = bvhPrimitives[i];
        }
        node->initLeaf(firstOffset, bvhPrimitives.size(), bounds);
        return node;
    } else {
        // Chose split dimensions
        AABB centroidBounds;
        for (const auto &prim: bvhPrimitives) {
            centroidBounds.expand(prim.centroid());
        }
        int dim = centroidBounds.longestAxis();

        if (centroidBounds.pmin[dim] == centroidBounds.pmax[dim]) {
            // CASE: empty bbox
            const int firstOffset = *orderedPrimitiveOffset;
            *orderedPrimitiveOffset += bvhPrimitives.size();
            for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                // const int index                    = bvhPrimitives[i].index;
                orderedPrimitives[firstOffset + i] = bvhPrimitives[i];
            }
            node->initLeaf(firstOffset, bvhPrimitives.size(), bounds);
            return node;
        }

        int mid = bvhPrimitives.size() / 2;

        if (bvhPrimitives.size() == 2) {
            std::nth_element(
                    bvhPrimitives.begin(),
                    bvhPrimitives.begin() + mid,
                    bvhPrimitives.end(),
                    [dim](const Primitive &a, const Primitive &b) {
                        return a.centroid()[dim] < b.centroid()[dim];
                    });
        } else {
            // Setup buckets
            constexpr int BVH_NUM_BUCKETS = 12;
            BVH2Bucket buckets[BVH_NUM_BUCKETS];

            for (const auto &prim: bvhPrimitives) {
                int b = BVH_NUM_BUCKETS * centroidBounds.offset(prim.centroid())[dim];
                if (b == BVH_NUM_BUCKETS) b = BVH_NUM_BUCKETS - 1;
                buckets[b].count++;
                buckets[b].bounds.expand(prim.bounds);
            }

            // Setup bucket costs
            constexpr int BVH_NUM_SPLITS = BVH_NUM_BUCKETS - 1;
            float costs[BVH_NUM_SPLITS]  = {};

            // Forward pass
            int countBelow = 0;
            AABB boundsBelow;
            for (int i = 0; i < BVH_NUM_SPLITS; ++i) {
                countBelow += buckets[i].count;
                boundsBelow.expand(buckets[i].bounds);
                costs[i] += countBelow * boundsBelow.surfaceArea();
            }

            // Backwards pass
            int countAbove = 0;
            AABB boundsAbove;
            for (int i = BVH_NUM_BUCKETS - 1; i > 0; --i) {
                countAbove += buckets[i].count;
                boundsAbove.expand(buckets[i].bounds);
                costs[i - 1] += countAbove * boundsAbove.surfaceArea();
            }

            // Find split
            int minBucket = -1;
            float minCost = INF;
            for (int i = 0; i < BVH_NUM_SPLITS; ++i) {
                if (costs[i] < minCost) {
                    minCost   = costs[i];
                    minBucket = i;
                }
            }

            // Calculate split cost
            const float leafCost = bvhPrimitives.size();
            minCost              = 0.5f + minCost / bounds.surfaceArea();
            if (bvhPrimitives.size() > maxPrimsInNode || minCost < leafCost) {
                // Build interior node
                auto midIterator = std::partition(bvhPrimitives.begin(), bvhPrimitives.end(), [=](const Primitive &p) {
                    int b = BVH_NUM_BUCKETS * centroidBounds.offset(p.centroid())[dim];
                    if (b == BVH_NUM_BUCKETS) b = BVH_NUM_BUCKETS - 1;
                    return b <= minBucket;
                });
                mid              = midIterator - bvhPrimitives.begin();
            } else {
                // Build leaf node
                const int firstOffset = *orderedPrimitiveOffset;
                *orderedPrimitiveOffset += bvhPrimitives.size();
                for (size_t i = 0; i < bvhPrimitives.size(); ++i) {
                    // const int index                    = bvhPrimitives[i].index;
                    orderedPrimitives[firstOffset + i] = bvhPrimitives[i];
                }
                node->initLeaf(firstOffset, bvhPrimitives.size(), bounds);
                return node;
            }
        }

        BVH2Node *children[2];
        children[0] = buildBVH2Tree(bvhPrimitives.subspan(0, mid), totalNodes, orderedPrimitiveOffset, orderedPrimitives, maxPrimsInNode);
        children[1] = buildBVH2Tree(bvhPrimitives.subspan(mid), totalNodes, orderedPrimitiveOffset, orderedPrimitives, maxPrimsInNode);
        node->initBranch(dim, children[0], children[1]);
    }

    return node;
}

int flattenBVH2(const BVH2Node *node, LinearBVH2Node *nodes, int *offset) {
    LinearBVH2Node *linearNode = &nodes[*offset];
    linearNode->bbox          = node->bbox;
    const int nodeOffset      = (*offset)++;
    if (node->numPrimitives > 0) {
        linearNode->primitivesOffset = node->firstPrimOffset;
        linearNode->numPrimitives    = node->numPrimitives;
    } else {
        linearNode->axis          = node->splitAxis;
        linearNode->numPrimitives = 0;
        flattenBVH2(node->children[0], nodes, offset);
        linearNode->secondChildOffset = flattenBVH2(node->children[1], nodes, offset);
    }
    return nodeOffset;
}