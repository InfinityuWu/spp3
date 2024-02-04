#include "Algorithm.h"

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

BitmapImage perform_scheme (const BitmapImage& original_image, const Key::key_type& encryption_key, const Algorithm::EncryptionScheme& scheme) noexcept {
  // Copying of data. Required or are we allowed to manipulate input data? This section could be easily parallelized
  const auto height = original_image.get_height();
  const auto width = original_image.get_width();
  auto result_image = BitmapImage{height, width};
  for (auto y = std::uint32_t(0); y < height; y++) {
    for (auto x = std::uint32_t(0); x < width; x++) {
      result_image.set_pixel(y, x, original_image.get_pixel(y, x));
    }
  }
  const key_type current_key{};
  for (int index = 0; index < 48; index++) {
    current_key[i] = encryption_key[i];
  }
  // Actual calculations
  for (int index = 0; index < 16; i++) {
    switch (scheme[index]) {
      case Algorithm::EncryptionStep::E:
        result_image = FES::encrypt(result_image, current_key);
        break;
      case Algorithm::EncryptionStep::D:
        result_image = FES::decrypt(result_image, current_key);
        break;
      case Algorithm::EncryptionStep::K:
        current_key = Key::produce_new_key(current_key);
        break;
      case Algorithm::EncryptionStep::T:
        result_image = result_image.transpose();
        break;
    }
    return result_image;
  }
}
