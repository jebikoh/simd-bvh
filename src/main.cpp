#include "bvh2.hpp"
#include "scene.hpp"
#include "simd.hpp"
#include "tests.hpp"

int main() {
#ifdef RUN_TESTS
    std::cout << "Running tests" << std::endl;
    for (std::size_t i = 0; i < TEST_FN_PTRS_SIZE; ++i) {
        TEST_FN_PTRS[i]();
    }
    std::cout << "All tests passed" << std::endl;
#endif

    Scene scene;
    scene.loadMesh("../src/assets/shaderball_hsd.obj");

    BVH2 bvh2{
            .maxPrimsInNode = 1,
            .scene          = scene};
    bvh2.build();

    bvh2.destroy();
    scene.destroy();
}
