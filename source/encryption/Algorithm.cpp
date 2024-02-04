#include "Algorithm.h"

std::uint64_t Algorithm::encode (const Algorithm::EncryptionScheme& scheme) noexcept {
  std::uint64_t encoded{0};
  for (int i = 0; i < 16; i++) {
    std::uint64_t addend = static_cast<std::uint64_t>(scheme[i]);
    /*
      // this switch is the alternative solution without utilizing associated values within the enum and just assigning them here accordingly
      std::uint64_t addend{0};
      switch (scheme[i]) {
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
    addend <<= 2 * i;
    encoded += addend;
  }
  return encoded;
}
