#pragma once

#include "common.hpp"
#include "aabb.hpp"
#include "primitives.hpp"

#include <complex>

struct Material;

struct SurfaceIntersection {
    Vec3f point;
    Vec3f normal;
    Vec2f uv;

    float t;
    bool frontFace;

    void setFaceNormal(const Ray &r, const Vec3f &n) {
        frontFace = jtx::dot(r.dir, n) < 0;
        normal    = frontFace ? n : -n;
    }
};

struct Mesh {
    int numVertices;
    int numIndices;

    Vec3i *indices;
    Vec3f *vertices;
    Vec3f *normals;

    Vec2f *uvs;

    void getVertices(const int index, Vec3f &v0, Vec3f &v1, Vec3f &v2) const {
        const Vec3i i = indices[index];

        v0 = vertices[i[0]];
        v1 = vertices[i[1]];
        v2 = vertices[i[2]];
    }

    [[nodiscard]]
    AABB tBounds(const int index) const {
        Vec3f v0, v1, v2;
        getVertices(index, v0, v1, v2);

        return AABB{v0, v1}.expand(v2);
    }

    [[nodiscard]]
    float tArea(const int index) const {
        Vec3f v0, v1, v2;
        getVertices(index, v0, v1, v2);
        return 0.5f * jtx::cross(v1 - v0, v2 - v0).len();
    }

    void getNormals(const int index, Vec3f &n0, Vec3f &n1, Vec3f &n2) const {
        const Vec3i i = indices[index];
        n0            = normals[i[0]];
        n1            = normals[i[1]];
        n2            = normals[i[2]];
    }

    void getUVs(const int index, Vec2f &uv0, Vec2f &uv1, Vec2f &uv2) const {
        const Vec3i i = indices[index];
        uv0           = uvs[i[0]];
        uv1           = uvs[i[1]];
        uv2           = uvs[i[2]];
    }

    bool tClosestHit(const Ray &r, const Interval t, SurfaceIntersection &record, const int index, float &b1, float &b2) const {
        Vec3f v0, v1, v2;
        getVertices(index, v0, v1, v2);
        const auto v0v1 = v1 - v0;
        const auto v0v2 = v2 - v0;
        const auto pvec = jtx::cross(r.dir, v0v2);
        const auto det  = v0v1.dot(pvec);

        if (fabs(det) < 1e-8) return false;

        const float invDet = 1 / det;
        const auto tvec    = r.origin - v0;

        b1 = tvec.dot(pvec) * invDet;
        if (b1 < 0 || b1 > 1) return false;

        const auto qvec = tvec.cross(v0v1);
        b2               = r.dir.dot(qvec) * invDet;
        if (b2 < 0 || b1 + b2 > 1) return false;

        const float root = v0v2.dot(qvec) * invDet;
        if (!t.surrounds(root)) return false;

        record.t        = root;
        record.point    = r.at(root);

        Vec3f n0, n1, n2;
        getNormals(index, n0, n1, n2);

        float b0 = (1 - b1 - b2);

        // Shading normal
        const Vec3f n = b0 * n0 + b1 * n1 + b2 * n2;
        record.setFaceNormal(r, n);

        // Interpolate UV
        Vec2f uv0, uv1, uv2;
        getUVs(index, uv0, uv1, uv2);
        record.uv = uv0 * b0 + uv1 * b1 + uv2 * b2;

        return true;
    }

    [[nodiscard]]
    bool tAnyHit(const Ray &r, const Interval t, const int index) const {
        Vec3f v0, v1, v2;
        getVertices(index, v0, v1, v2);
        const auto v0v1 = v1 - v0;
        const auto v0v2 = v2 - v0;
        const auto pvec = jtx::cross(r.dir, v0v2);
        const auto det  = v0v1.dot(pvec);

        if (fabs(det) < 1e-8) return false;

        const float invDet = 1 / det;
        const auto tvec    = r.origin - v0;

        const auto u = tvec.dot(pvec) * invDet;
        if (u < 0 || u > 1) return false;

        const auto qvec = tvec.cross(v0v1);
        const auto v    = r.dir.dot(qvec) * invDet;
        if (v < 0 || u + v > 1) return false;

        const float root = v0v2.dot(qvec) * invDet;
        if (!t.surrounds(root)) return false;

        return true;
    }

    void destroy() const {
        if (indices) delete[] indices;
        if (vertices) delete[] vertices;
        if (normals) delete[] normals;
        if (uvs) delete[] uvs;
    }
};