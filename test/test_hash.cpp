#include "test.h"
#include "test_declarations.h"

#include "cuda/common.cuh"
#include "util/Hash.h"

#include <cstdint>
#include <vector>

class HashTest : public LabTest {};

TEST_F(HashTest, test_three_a_one) {}

TEST_F(HashTest, test_three_a_two) {}

TEST_F(HashTest, test_three_b_one) {
	const auto test_kernel = [](const auto number_values, const auto number_threads) {
		auto values = std::vector<std::uint64_t>(number_values);
		auto hashes_cpu = std::vector<std::uint64_t>(number_values);
		auto hashes_gpu = std::vector<std::uint64_t>(number_values);

		for (auto i = 0; i < number_values; i++) {
			const auto proto_value = static_cast<std::uint64_t>(i) + 137 * number_threads + (number_threads % 13);
			const auto value = Hash::hash(proto_value);
			const auto hash = Hash::hash(value);

			values[i] = value;
			hashes_cpu[i] = hash;
		}

		auto valid = test_three_b_adapter(values.data(), hashes_gpu.data(), number_values, number_threads);
		ASSERT_TRUE(valid);

		for (auto i = 0; i < number_values; i++) {
			const auto cpu_value = hashes_cpu[i];
			const auto gpu_value = hashes_gpu[i];

			ASSERT_EQ(cpu_value, gpu_value);
		}
		};

	const auto number_values_1 = 1024U;
	const auto number_threads_1 = 1024U;

	test_kernel(number_values_1, number_threads_1);

	const auto number_values_2 = 512U;
	const auto number_threads_2 = 1024U;

	test_kernel(number_values_2, number_threads_2);

	const auto number_values_3 = 2048U;
	const auto number_threads_3 = 64U;

	test_kernel(number_values_3, number_threads_3);
}

TEST_F(HashTest, test_three_b_two) {
	const auto test_kernel = [](const auto number_values, const auto number_threads) {
		auto values = std::vector<std::uint64_t>(number_values);
		auto hashes_cpu = std::vector<std::uint64_t>(number_values);
		auto hashes_gpu = std::vector<std::uint64_t>(number_values);

		for (auto i = 0; i < number_values; i++) {
			const auto proto_value = static_cast<std::uint64_t>(i) + 137 * number_threads + (number_threads % 13);
			const auto value = Hash::hash(proto_value);
			const auto hash = Hash::hash(value);

			values[i] = value;
			hashes_cpu[i] = hash;
		}

		auto valid = test_three_b_adapter(values.data(), hashes_gpu.data(), number_values, number_threads);
		ASSERT_TRUE(valid);

		for (auto i = 0; i < number_values; i++) {
			const auto cpu_value = hashes_cpu[i];
			const auto gpu_value = hashes_gpu[i];

			ASSERT_EQ(cpu_value, gpu_value);
		}
		};

	const auto number_values_1 = 1024U;
	const auto number_threads_1 = 37U;

	test_kernel(number_values_1, number_threads_1);

	const auto number_values_2 = 1025U;
	const auto number_threads_2 = 32U;

	test_kernel(number_values_2, number_threads_2);

	const auto number_values_3 = 10000U;
	const auto number_threads_3 = 111U;

	test_kernel(number_values_3, number_threads_3);
}

TEST_F(HashTest, test_three_c_one) {}

TEST_F(HashTest, test_three_c_two) {}

TEST_F(HashTest, test_three_d_one) {}

TEST_F(HashTest, test_three_d_two) {}

TEST_F(HashTest, test_three_e_one) {}

TEST_F(HashTest, test_three_e_two) {}
