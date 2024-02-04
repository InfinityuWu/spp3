#include "Algorithm.h"

#include <exception>

std::uint64_t Algorithm::encode (const Algorithm::EncryptionScheme& scheme) noexcept {
  std::uint64_t encoded{0};
  for (int index = 0; index < 16; index++) {
    std::uint64_t addend = static_cast<std::uint64_t>(scheme[index]);
    /*
      // this switch is the alternative solution without utilizing associated values within the enum and just assigning them here accordingly
      std::uint64_t addend{0};
      switch (scheme[index]) {
        case Algorithm::EncryptionStep::E:
          break;
        case Algorithm::EncryptionStep::D:
          addend = 1;
          break;
        case Algorithm::EncryptionStep::K:
          addend = 2;
          break;
        case Algorithm::EncryptionStep::T:
          addend = 3;
          break;
      }
    */
    addend <<= 2 * index;
    encoded += addend;
  }
  encoded += encoded << 32;
  return encoded;
}

Algorithm::EncryptionScheme Algorithm::decode (const std::uint64_t encoded) {
  Algorithm::EncryptionScheme scheme{};
  for (int index = 0; index < 16; index++) {
    uint64_t first_appearance = (encoded << (62 - 2 * index)) >> 62;
    uint64_t second_appearance = (encoded << (30 - 2 * index)) >> 62;
    if (first_appearance != second_appearance) throw std::exception{};
    if (first_appearance == 0) scheme[index] = Algorithm::EncryptionStep::E;
    else if (first_appearance == 1) scheme[index] = Algorithm::EncryptionStep::D;
    else if (first_appearance == 2) scheme[index] = Algorithm::EncryptionStep::K;
    else if (first_appearance == 3) scheme[index] = Algorithm::EncryptionStep::T;
    else throw std::exception{};
  }
  return scheme;
}
