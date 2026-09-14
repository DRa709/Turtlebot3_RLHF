import unittest

from turtlebot3_drl_nav.validator import _i


class ValidatorIntegerParsingTests(unittest.TestCase):
    def test_sha256_derived_63_bit_seed_is_exact(self):
        seed = 6678464068980594013
        self.assertGreater(seed, 2 ** 53)
        self.assertEqual(_i(str(seed)), seed)

    def test_integer_parser_does_not_accept_float_syntax_or_whitespace(self):
        for value in ("1.0", "1e3", "nan", "inf", " 1", "1 ", "+", "-"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _i(value)

    def test_empty_field_remains_not_applicable(self):
        self.assertIsNone(_i(""))


if __name__ == "__main__":
    unittest.main()
