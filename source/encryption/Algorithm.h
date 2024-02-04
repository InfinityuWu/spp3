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

  private:

};
