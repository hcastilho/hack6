import pytest

from webapp import webapp


@pytest.fixture
def app():
    return webapp.app