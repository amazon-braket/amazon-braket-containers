import os
import pytest


def pytest_addoption(parser):
    parser.addoption('--region', default=os.getenv("REGION"))
    parser.addoption('--account', default=os.getenv("ACCOUNT_ID"))
    parser.addoption('--repository', default=os.getenv("REPOSITORY_NAME"))
    parser.addoption('--role', required=True)
    parser.addoption('--tag', required=True)


@pytest.fixture(scope='session')
def region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session')
def account(request):
    return request.config.getoption('--account')


@pytest.fixture(scope='session')
def repository_name(request):
    return request.config.getoption('--repository')


@pytest.fixture(scope='session')
def role(request):
    return request.config.getoption('--role')


@pytest.fixture(scope='session')
def image_tag(request):
    return request.config.getoption('--tag')

