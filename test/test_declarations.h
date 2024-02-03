#pragma once

#include "image/pixel.h"

#include <cstdint>

bool test_two_b_adapter(const Pixel<std::uint8_t>* const input, Pixel<std::uint8_t>* const output, const unsigned int width, const unsigned int height);

bool test_three_b_adapter(const std::uint64_t* const values, std::uint64_t* const hashes, const unsigned int length, const unsigned int number_threads);
