#include "bvh2.hpp"
#include "scene.hpp"
#include "simd.hpp"

int main() {
111111111111111111111111111


    Scene scene;
    scene.loadMesh("../src/assets/shaderball_hsd.obj");

    BVH2 bvh2{
            .maxPrimsInNode = 1,
            .scene          = scene};
    bvh2.build();

    bvh2.destroy();
    scene.destroy();
}
