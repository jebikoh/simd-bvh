#pragma once

#include "common.hpp"
#include "aabb.hpp"

struct Triangle {
    int index;
    int meshIndex;
};

struct Primitive {
    enum Type {
        SPHERE = 0,
        TRIANGLE = 1,
    };

    Type type;
    size_t index;
    AABB bounds;

    [[nodiscard]]
    Vec3f centroid() const {
        return 0.5f * bounds.pmin + 0.5f * bounds.pmax;
    }
};