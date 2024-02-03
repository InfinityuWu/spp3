#include "test.h"
#include "test_declarations.h"

#include "cuda/common.cuh"
#include "cuda/image.cuh"
#include "image/bitmap_image.h"

#include <vector>

class GrayTest : public LabTest {};

TEST_F(GrayTest, test_two_a_one) {
	const auto test_values = [](const unsigned int numerator, const unsigned int denominator, const unsigned int expected) {
		const auto val = divup(numerator, denominator);
		ASSERT_EQ(val, expected);
		};

	test_values(32, 1, 32);
	test_values(64, 64, 1);
	test_values(1024 * 1024 * 1024, 1024 * 32, 1024 * 32);
	test_values(1025, 32, 33);
	test_values(1024, 32, 32);
	test_values(1023, 32, 32);
	test_values(1055, 32, 33);
}

TEST_F(GrayTest, test_two_b_one) {
	const auto construct_pixel = [](const std::uint8_t red, const std::uint8_t green, const std::uint8_t blue) -> BitmapImage::BitmapPixel {
		switch (BitmapImage::BitmapPixel::channel_order) {
		case ChannelOrder::BGR:
			return { blue, green, red };
		case ChannelOrder::BRG:
			return { blue, red, green };
		case ChannelOrder::GBR:
			return { green, blue, red };
		case ChannelOrder::GRB:
			return { green, red, blue };
		case ChannelOrder::RBG:
			return { red, blue, green };
		case ChannelOrder::RGB:
			return { red, green, blue };
		default:
			EXPECT_TRUE(false) << "ChannelOrder for BitmapImage::BitmapPixel has no valid value!\n";
		}

		return {};
		};

	const auto test_kernel = [construct_pixel](const auto width, const auto height) {
		auto pixels_color = std::vector<BitmapImage::BitmapPixel>(width * height);
		auto pixels_gray_cpu = std::vector<BitmapImage::BitmapPixel>(width * height);
		auto pixels_gray_gpu = std::vector<BitmapImage::BitmapPixel>(width * height);

		for (auto y = 0; y < height; y++) {
			for (auto x = 0; x < width; x++) {
				const auto r = static_cast<std::uint8_t>((((width % 109) + 1) * (x + 3)) % 256);
				const auto g = static_cast<std::uint8_t>((((height % 73) + 3) * (x + y + 8)) % 256);
				const auto b = static_cast<std::uint8_t>((((width % 23) + 7) * (y + 17)) % 256);

				const auto pixel = construct_pixel(r, g, b);

				pixels_color[y * width + x] = pixel;

				const auto gray = r * 0.2989 + g * 0.5870 + b * 0.1140;
				const auto gray_converted = static_cast<std::uint8_t>(gray);

				const auto gray_pixel = BitmapImage::BitmapPixel{ gray_converted , gray_converted,  gray_converted };

				pixels_gray_cpu[y * width + x] = gray_pixel;
			}
		}

		auto valid = test_two_b_adapter(static_cast<const std::vector<BitmapImage::BitmapPixel>&>(pixels_color).data(), pixels_gray_gpu.data(), width, height);
		ASSERT_TRUE(valid);

		for (auto y = 0; y < height; y++) {
			for (auto x = 0; x < width; x++) {
				const auto& calculated_pixel = pixels_gray_gpu[y * width + x];
				const auto r = calculated_pixel.get_red_channel();
				const auto g = calculated_pixel.get_green_channel();
				const auto b = calculated_pixel.get_blue_channel();

				ASSERT_EQ(r, g);
				ASSERT_EQ(r, b);

				const auto gray_value = pixels_gray_cpu[y * width + x].get_red_channel();

				auto diff = 0;
				if (gray_value > r) {
					diff = gray_value - r;
				}
				else {
					diff = r - gray_value;
				}

				ASSERT_LE(diff, 1);
			}
		}
		};

	const auto width_1 = 2048U;
	const auto height_1 = 1U;

	test_kernel(width_1, height_1);

	const auto width_2 = 1U;
	const auto height_2 = 2048U;

	test_kernel(width_2, height_2);

	const auto width_3 = 1024U;
	const auto height_3 = 1024U;

	test_kernel(width_3, height_3);
}

TEST_F(GrayTest, test_two_b_two) {}

TEST_F(GrayTest, test_two_c_one) {}

TEST_F(GrayTest, test_two_c_two) {}
