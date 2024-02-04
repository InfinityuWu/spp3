#pragma once
#include <array>
#include <cstdint>

class Algorithm {

  public:

    // Task 1a)
    enum EncryptionStep {E, D, K, T};                        // leave at class scope or put at file scope maybe?
    using EncryptionScheme = std::array<EncryptionStep,16>;  // alternatively use typedef instead of using for alias

    // Task 1b)
    [[nodiscard]] static std::uint64_t encode (const EncryptionScheme& scheme) noexcept;

    // Task 1c)
    [[nodiscard]] static EncryptionScheme decode (const std::uint64_t encoded);

    // Task 1d)
    [[nodiscard]] static BitmapImage perform_scheme (const BitmapImage& original_image, const FES::key_type& encryption_key, const EncryptionScheme& scheme) noexcept;

  private:

};
