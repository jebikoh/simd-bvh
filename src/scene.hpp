#pragma once

#include "common.hpp"
#include "mesh.hpp"
#include "primitives.hpp"

struct CameraProperties {
    Vec3f center;
    Vec3f target;
    Vec3f up;
    float yfov;
    float defocusAngle;
    float focusDistance;
};

struct Material {
    enum Type {
        DIFFUSE = 0,
        DIELECTRIC = 1,
        CONDUCTOR = 2
    };

    Type type;
    Vec3f albedo;
    float refractionIndex;
    Vec3f IOR;
    Vec3f k;
    float alphaX, alphaY;
    Vec3f emission = Vec3f(0, 0, 0);

    int texId;
};


struct Scene {
    std::string name;

    std::vector<Material> materials;

    std::vector<Triangle> triangles;
    std::vector<Mesh> meshes;

    void loadMesh(const std::string &path);

    int numPrimitives() const {
        return triangles.size();
    }

    void destroy() const {
        for (auto &mesh : meshes) {
            mesh.destroy();
        }
    }
};
