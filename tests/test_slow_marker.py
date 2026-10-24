"""The `slow` marker set up in conftest.py: marked tests run only with RUN_SLOW=1, and a
test that merely has "slow" among its keywords (a parametrize id, say) is not skipped."""

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = (Path(__file__).resolve().parent.parent / "conftest.py").read_text()
TESTS = """
import pytest

@pytest.mark.slow
def test_trains():
    pass

@pytest.mark.parametrize("speed", ["slow", "fast"])
def test_speed(speed):
    pass
"""


@pytest.mark.parametrize("run_slow, passed, skipped", [(None, 2, 1), ("1", 3, 0)])
def test_only_marked_tests_are_slow(pytester, monkeypatch, run_slow, passed, skipped):
    if run_slow is None:
        monkeypatch.delenv("RUN_SLOW", raising=False)
    else:
        monkeypatch.setenv("RUN_SLOW", run_slow)
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(TESTS)
    pytester.runpytest("-q", "-p", "no:cacheprovider").assert_outcomes(passed=passed, skipped=skipped)


def test_the_marker_is_registered(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(TESTS)
    pytester.runpytest("-q", "--strict-markers", "-p", "no:cacheprovider").assert_outcomes(passed=2, skipped=1)
