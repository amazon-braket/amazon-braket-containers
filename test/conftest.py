# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

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

