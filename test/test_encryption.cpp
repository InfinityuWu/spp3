#include "test.h"

#include "encryption/Algorithm.h"

class EncryptionTest : public LabTest {};

TEST_F(EncryptionTest, test_one_a_one) {
}

TEST_F(EncryptionTest, test_one_b_one) {
	auto scheme_1 = EncryptionScheme{};
	scheme_1[0] = EncryptionStep::E;
	scheme_1[1] = EncryptionStep::D;
	scheme_1[2] = EncryptionStep::D;
	scheme_1[3] = EncryptionStep::K;
	scheme_1[4] = EncryptionStep::K;
	scheme_1[5] = EncryptionStep::T;
	scheme_1[6] = EncryptionStep::E;
	scheme_1[7] = EncryptionStep::T;
	scheme_1[8] = EncryptionStep::T;
	scheme_1[9] = EncryptionStep::T;
	scheme_1[10] = EncryptionStep::K;
	scheme_1[11] = EncryptionStep::K;
	scheme_1[12] = EncryptionStep::E;
	scheme_1[13] = EncryptionStep::D;
	scheme_1[14] = EncryptionStep::E;
	scheme_1[15] = EncryptionStep::D;

	const auto expected_code_1 = 0b01000100'10101111'11001110'10010100'01000100'10101111'11001110'10010100ULL;
	const auto code_1 = encode(scheme_1);

	ASSERT_EQ(code_1, expected_code_1);

	auto scheme_2 = EncryptionScheme{};
	scheme_2[0] = EncryptionStep::K;
	scheme_2[1] = EncryptionStep::T;
	scheme_2[2] = EncryptionStep::K;
	scheme_2[3] = EncryptionStep::T;
	scheme_2[4] = EncryptionStep::E;
	scheme_2[5] = EncryptionStep::D;
	scheme_2[6] = EncryptionStep::E;
	scheme_2[7] = EncryptionStep::D;
	scheme_2[8] = EncryptionStep::K;
	scheme_2[9] = EncryptionStep::T;
	scheme_2[10] = EncryptionStep::K;
	scheme_2[11] = EncryptionStep::T;
	scheme_2[12] = EncryptionStep::E;
	scheme_2[13] = EncryptionStep::D;
	scheme_2[14] = EncryptionStep::E;
	scheme_2[15] = EncryptionStep::D;

	const auto expected_code_2 = 0b01000100'11101110'01000100'11101110'01000100'11101110'01000100'11101110ULL;
	const auto code_2 = encode(scheme_2);

	ASSERT_EQ(code_2, expected_code_2);
}

TEST_F(EncryptionTest, test_one_c_one) {
	
}

TEST_F(EncryptionTest, test_one_c_two) {
}

TEST_F(EncryptionTest, test_one_d_one) {
	
}
