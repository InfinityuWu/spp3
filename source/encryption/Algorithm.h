#pragma once

#include <array>
#include <cstdint>
#include <exception>

#include "encryption/FES.h"
#include "encryption/Key.h"
#include "image/bitmap_image.h"

class Algorithm {

  public:

    // Task 1a)
    enum class EncryptionStep : std::uint64_t {E = 0, D = 1, K = 2, T = 3};  // leave at class scope or put at file scope maybe?
    using EncryptionScheme = std::array<EncryptionStep,16>;                  // alternatively use typedef instead of using for alias

    // Task 1b)
    [[nodiscard]] static std::uint64_t encode (const EncryptionScheme& scheme) noexcept;

    // Task 1c)
    [[nodiscard]] static EncryptionScheme decode (const std::uint64_t encoded);

    // Task 1d)
    [[nodiscard]] static BitmapImage perform_scheme (const BitmapImage& original_image, const Key::key_type& encryption_key, const EncryptionScheme& scheme) noexcept;

  private:

};
