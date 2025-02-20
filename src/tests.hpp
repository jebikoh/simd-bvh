#pragma once

#include "bvh4.hpp"

#include <vector>

using TestFnPtr = void (*)();
extern const TestFnPtr TEST_FN_PTRS[];
extern const std::size_t TEST_FN_PTRS_SIZE;
