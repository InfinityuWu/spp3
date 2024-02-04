#include "Algorithm.h"

std::uint64_t Algorithm::encode (const Algorithm::EncryptionScheme& scheme) noexcept {
  std::uint64_t encoded{0};
  for (int i = 0; i < 16; i++) {
    std::uint64_t addend{0};
    // this switch could be replaced by utilizing associated values within the enum and just assigning them here accordingly
    switch (scheme[i]) {
      case E:
        break;
      case D:
        addend = 1;
        break;
      case K:
        addend = 2;
        break;
      case T:
        addend = 3;
        break;
    }
    addend <<= 2 * i;
    encoded += addend;
  }
  return encoded;
}
