#pragma once

#include "common.hpp"

struct Interval {
    float min, max;

    Interval()
        : min(INF),
          max(-INF) {}

    Interval(const float min, const float max)
        : min(min),
          max(max) {}

    [[nodiscard]]
    float size() const {
        return max - min;
    }

    [[nodiscard]]
    bool contains(const float x) const {
        return min <= x && x <= max;
    }

    [[nodiscard]]
    bool surrounds(const float x) const {
        return min < x && x < max;
    }

    [[nodiscard]]
    float clamp(const float x) const {
        return jtx::clamp(x, min, max);
    }

    [[nodiscard]]
    Interval expand(const float delta) const {
        const auto padding = delta / 2;
        return {min - padding, max + padding};
    }
};

// https://github.com/jebikoh/JTX-PathTracer/blob/MIS/src/util/aabb.hpp
struct AABB {
    Vec3f pmin, pmax;

    AABB() {
        float minNum = std::numeric_limits<float>::lowest();
        float maxNum = std::numeric_limits<float>::max();
        pmin         = {maxNum, maxNum, maxNum};
        pmax         = {minNum, minNum, minNum};
    }

    AABB(const Vec3f &a, const Vec3f &b) {
        pmin = jtx::min(a, b);
        pmax = jtx::max(a, b);
    }

    AABB(const Interval &x, const Interval &y, const Interval &z) {
        pmin = {x.min, y.min, z.min};
        pmax = {x.max, y.max, z.max};
    }

    AABB(const AABB &a, const AABB &b) {
        pmin = jtx::min(a.pmin, b.pmin);
        pmax = jtx::max(a.pmax, b.pmax);
    }

    AABB expand(const AABB &other) {
        pmin = jtx::min(pmin, other.pmin);
        pmax = jtx::max(pmax, other.pmax);
        return *this;
    }

    AABB expand(const Vec3f &p) {
        pmin = jtx::min(pmin, p);
        pmax = jtx::max(pmax, p);
        return *this;
    }

    [[nodiscard]] Interval axis(const int i) const {
        if (i == 0)
            return {pmin.x, pmax.x};
        if (i == 1)
            return {pmin.y, pmax.y};
        return {pmin.z, pmax.z};
    }

    [[nodiscard]] int longestAxis() const {
        const Vec3f d = diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        if (d.y > d.z)
            return 1;
        return 2;
    }

    [[nodiscard]]
    Vec3f offset(const Vec3f &p) const {
        Vec3f o = p - pmin;
        if (pmax.x > pmin.x)
            o.x /= pmax.x - pmin.x;
        if (pmax.y > pmin.y)
            o.y /= pmax.y - pmin.y;
        if (pmax.z > pmin.z)
            o.z /= pmax.z - pmin.z;
        return o;
    }

    [[nodiscard]] bool hit(const Vec3f &o, const Vec3f &d, const Interval &t) const {
        auto t0 = t.min;
        auto t1 = t.max;

        for (int i = 0; i < 3; ++i) {
            const auto invDir = 1 / d[i];
            auto tNear        = (pmin[i] - o[i]) * invDir;
            auto tFar         = (pmax[i] - o[i]) * invDir;

            if (tNear > tFar)
                std::swap(tNear, tFar);
            t0 = tNear > t0 ? tNear : t0;
            t1 = tFar < t1 ? tFar : t1;
            if (t0 > t1)
                return false;
        }
        return true;
    }

    [[nodiscard]]
    Vec3f diagonal() const { return pmax - pmin; }

    [[nodiscard]]
    float surfaceArea() const {
        const Vec3f d = diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    [[nodiscard]]
    float volume() const {
        const Vec3f d = diagonal();
        return d.x * d.y * d.z;
    }
};
