#pragma once

#include <array>
#include <cstdint>
#include <exception>
#include <limits>

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

    // Task 3e)
    [[nodiscard]] static EncryptionScheme retrieve_scheme(unsigned std::uint64_t hash){
        // hash der Kodierung eines Schemas
        // 010203040506070809101112131415
        //  ? ? ? ? ? ? ? ? ? E E E E E E

        // c = encode(decode(c)); c := Zahl
        // s = decode(encode(s)); s_:= EncryptionScheme

        // c = hash(encode(s)) && s = retrieve_scheme(c)
        // ==> s = retrieve_scheme(hash(encode(s)));
        // s = s = decode(dehash(hash(encode(s))));

        constexpr auto val_a = std::uint64_t{ 5'647'095'006'226'412'969 };
        constexpr auto val_b = std::uint64_t{ 41'413'938'183'913'153 };
        constexpr auto val_c = std::uint64_t{ 6'225'658'194'131'981'369 };
        std::uint64_t correctValue;
#pragma omp parallel for lastPrivate(correctValue)
        for (std::uint32_t i = 0; i <= std::numeric_limits<std::uint32_t>::max(); i++){
            std::uint64_t value = i;
            value += value << 32;
            const auto val_1 = (value >> 14) + val_a;
            const auto val_2 = (value << 54) ^ val_b;
            const auto val_3 = (val_1 + val_2) << 4;
            const auto val_4 = (value % val_c) * 137;
            const auto final_hash = val_3 ^ val_4;
            if (final_hash == hash){
                correctValue = value;
            }
        }
        return decode(correctValue);
    }
  private:

};
