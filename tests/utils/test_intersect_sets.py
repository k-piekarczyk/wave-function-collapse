import pytest

from wave_function_collapse.utils.set_intersection import set_intersection


@pytest.mark.parametrize(
    "set_1, set_2, expected_result",
    [
        [{1, 2, 3}, {3, 4, 5}, {3}],
        [{"test", "toast", "ghost"}, {"boast", "ghost", "dough"}, {"ghost"}],
    ],
)
def test_should_intersect_two_sets(set_1, set_2, expected_result):
    # when
    result = set_intersection(set_1=set_1, set_2=set_2)

    # then
    assert result == expected_result
