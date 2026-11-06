import pytest

from training.utils.expect import expect


def test_expect_returns_a_present_value_unchanged():
    assert expect(value=0, message="unused") == 0


def test_expect_raises_the_message_on_none():
    with pytest.raises(expected_exception=ValueError, match="draw one first"):
        expect(value=None, message="draw one first")
