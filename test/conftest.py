import pytest
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def pytest_addoption(parser):
    # Optional command line argument to specify one or several image sizes.
    parser.addoption(
        "--N", type=int, nargs='+', action="store", default=[192], help="Image size (default: 192)"
    )
    parser.addini(
        'datafile', 'URL to small ROOT data file for testing.', type="linelist"
    )


def pytest_generate_tests(metafunc):
    # If a test requires the fixture N, use the command line option.
    if 'N' in metafunc.fixturenames:
        metafunc.parametrize("N",
                             metafunc.config.getoption('--N'))
    if 'datafile' in metafunc.fixturenames:
        metafunc.parametrize("datafile",
                             metafunc.config.getini('datafile'))


@pytest.fixture
def data(tmp_path, datafile):
    """
    Downloading the datafile here will cache it once and for all.
    """
    import urllib
    filename = 'test'
    datafile_url = datafile
    data_path = os.path.join(tmp_path, filename + '.root')
    urllib.request.urlretrieve(datafile_url, data_path)
    return filename
