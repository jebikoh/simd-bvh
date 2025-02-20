#include "scene.hpp"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

void Scene::loadMesh(const std::string &path) {
    int totalVertices = 0;
    int totalFaces = 0;
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
    if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
        std::cerr << "Assimp error: " << importer.GetErrorString() << std::endl;
        return;
    }

    for (unsigned int m = 0; m < scene->mNumMeshes; m++) {
        aiMesh *aiMeshPtr = scene->mMeshes[m];

        size_t numVerts   = aiMeshPtr->mNumVertices;
        auto finalVerts   = new Vec3f[numVerts];
        auto finalNormals = new Vec3f[numVerts];

        Vec2f *finalUVs = nullptr;
        bool hasUV      = (aiMeshPtr->mTextureCoords[0] != nullptr);
        if (hasUV) {
            finalUVs = new Vec2f[numVerts];
        }

        for (size_t i = 0; i < numVerts; i++) {
            aiVector3D v  = aiMeshPtr->mVertices[i];
            finalVerts[i] = Vec3f(v.x, v.y, v.z);

            if (aiMeshPtr->HasNormals()) {
                aiVector3D n    = aiMeshPtr->mNormals[i];
                finalNormals[i] = Vec3f(n.x, n.y, n.z);
            } else {
                finalNormals[i] = Vec3f(0.0f, 1.0f, 0.0f);
            }

            if (hasUV) {
                aiVector3D uv = aiMeshPtr->mTextureCoords[0][i];
                finalUVs[i]   = Vec2f(uv.x, uv.y);
            } else if (finalUVs) {
                finalUVs[i] = Vec2f(0.0f, 0.0f);
            }
        }

        size_t numTriangles = aiMeshPtr->mNumFaces;
        auto *finalIndices  = new Vec3i[numTriangles];
        for (size_t i = 0; i < numTriangles; i++) {
            aiFace face = aiMeshPtr->mFaces[i];
            if (face.mNumIndices != 3) {
                std::cerr << "Warning: mesh " << m << " has a face that isn't a triangle.\n";
                continue;
            }
            finalIndices[i] = Vec3i(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
        }

        meshes.push_back(Mesh{static_cast<int>(numVerts), static_cast<int>(numTriangles), finalIndices, finalVerts, finalNormals, finalUVs});

        const int meshIndex = static_cast<int>(meshes.size()) - 1;
        for (size_t t = 0; t < numTriangles; t++) {
            Triangle tri;
            tri.index     = static_cast<int>(t);
            tri.meshIndex = meshIndex;
            triangles.push_back(tri);
        }

        totalVertices += numVerts;
        totalFaces += numTriangles;
    }

    std::cout << "Loaded scene with " << meshes.size() << " meshes.\n";
    std::cout << "Total vertices: " << totalVertices << "\n";
    std::cout << "Total faces: " << totalFaces << "\n";
}
