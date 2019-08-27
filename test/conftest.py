import pytest


def pytest_addoption(parser):
    # Optional command line argument to specify one or several image sizes.
    parser.addoption(
        "--N", type=int, nargs='+', action="store", default=[192], help="Image size (default: 192)"
    )


# If a test requires the fixture N, use the command line option.
def pytest_generate_tests(metafunc):
    if 'N' in metafunc.fixturenames:
        metafunc.parametrize("N",
                             metafunc.config.getoption('--N'))
