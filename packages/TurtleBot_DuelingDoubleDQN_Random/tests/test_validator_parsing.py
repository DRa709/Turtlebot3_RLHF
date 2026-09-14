import unittest

from turtlebot3_drl_nav.validator import _i


class ValidatorIntegerParsingTests(unittest.TestCase):
    def test_sha256_derived_63_bit_seed_is_exact(self):
        values = (
            2 ** 53 - 1,
            2 ** 53,
            2 ** 63 - 1,
            6678464068980594013,
        )
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(_i(str(value)), value)

    def test_integer_parser_does_not_accept_float_syntax_or_whitespace(self):
        for value in ("1.0", "1e3", "nan", "inf", " 1", "1 ", "+", "-"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _i(value)

    def test_empty_field_remains_not_applicable(self):
        self.assertIsNone(_i(""))


if __name__ == "__main__":
    unittest.main()
