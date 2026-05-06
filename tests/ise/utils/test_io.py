import pytest

from ise.utils.io import check_type


class TestCheckType:
    def test_passes_for_correct_single_type(self):
        assert check_type(42, int) == 1

    def test_passes_for_correct_type_in_tuple(self):
        assert check_type("hello", (str, int)) == 1

    def test_passes_for_second_type_in_tuple(self):
        assert check_type(3, (str, int)) == 1

    def test_raises_for_wrong_type(self):
        with pytest.raises(TypeError):
            check_type(3.14, int)

    def test_raises_for_none_when_not_allowed(self):
        with pytest.raises(TypeError):
            check_type(None, int)

    def test_passes_for_none_type(self):
        assert check_type(None, type(None)) == 1

    def test_error_message_mentions_expected_type(self):
        with pytest.raises(TypeError, match="int"):
            check_type("oops", int)

    def test_subclass_passes(self):
        class MyList(list):
            pass

        assert check_type(MyList(), list) == 1
