import pytest
from parametrized import parametrized

from debug import init_debug_module

debug = init_debug_module()


def test_shorten_single_line_error():
    with pytest.raises(ValueError, match="must be >= 3"):
        debug.shorten("12345678", 2)


@pytest.mark.parametrize(
    "string, limit, expected, placeholder_len, char_count",
    [
        # No error even if limit is < 4 because the string is <= limit.
        ("12", 2, "12", 0, 2),
        ("12345678", 3, "1..", 2, 1),
        ("12345678", 4, "1..8", 2, 2),
        ("12345678", 5, "1...8", 3, 2),
        ("12345678", 6, "12...8", 3, 3),
        ("12345678", 7, "12...78", 3, 4),
        ("12345678", 8, "12345678", 0, 8),
        ("1234567890", 7, "12...90", 3, 4),
        ("1234567890", 8, "123...90", 3, 5),
        ("1234567890", 9, "123...890", 3, 6),
        ("12345678901", 10, "1234...901", 3, 7),
        ("123456789012", 10, "1234...012", 3, 7),
        ("123456789012", 11, "1234...9012", 3, 8),
    ],
)
def test_shorten_single_line_valid(string, limit, expected, placeholder_len, char_count):
    actual = debug.shorten(string, limit)
    assert actual == expected
    assert len(actual) == limit

    placeholder = "." * placeholder_len
    assert actual.count(".") == placeholder_len

    actual_chars = actual.replace(placeholder, "", 1)
    expected_chars = expected.replace(placeholder, "", 1)
    assert actual_chars == expected_chars
    assert len(actual_chars) == char_count


@pytest.mark.parametrize(
    "string, limit, expected, placeholder_len, char_count",
    [
        ("12", 5, "12", 0, 2),
    ],
)
def test_shorten_single_line_large_limit_returns_as_is(string, limit, expected, placeholder_len, char_count):
    actual: str = debug.shorten(string, limit)
    assert actual == expected
    assert "." not in actual
    assert len(actual) == char_count


cases = [
    dict(string="\n2345678", limit=3, expected=" .."),
    dict(string="123\n5678", limit=5, expected="1...8"),
    dict(string="1\n345678", limit=5, expected="1...8"),
    dict(string="1234567\n", limit=5, expected="1... "),
]

strings, limits, expecteds = list(zip(*[case.values() for case in cases]))


@parametrized.zip
def test_shorten_multiline_valid(string=strings, limit=limits, expected=expecteds):
    actual: str = debug.shorten(string, limit)
    assert actual == expected
    assert len(actual) == limit
    assert "\n" not in actual

    placeholder = "." * expected.count(".")
    assert placeholder in actual

    actual_chars = actual.replace(placeholder, "", 1)
    expected_chars = expected.replace(placeholder, "", 1)
    assert actual_chars == expected_chars


@pytest.mark.parametrize(
    "string, limit, expected, placeholder_len, char_count",
    [
        ("123\n5678", 8, "123\n5678", 0, 8),
    ],
)
def test_shorten_multiline_large_limit_returns_as_is(string, limit, expected, placeholder_len, char_count):
    actual: str = debug.shorten(string, limit)
    assert actual == expected
    assert "." not in actual
    assert len(actual) == char_count


def test_shorten_multiline_error():
    with pytest.raises(ValueError, match="must be >= 3"):
        debug.shorten("123\n5678", 2)
