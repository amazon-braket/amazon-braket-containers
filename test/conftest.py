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
import json


def pytest_addoption(parser):
    parser.addoption('--region', default=os.getenv("REGION"))
    parser.addoption('--account', default=os.getenv("ACCOUNT_ID"))
    parser.addoption('--repository', default=os.getenv("REPOSITORY_NAME"))
    parser.addoption('--from-build-results', default=os.getenv("BUILD_RESULTS_PATH"))
    parser.addoption('--s3-bucket', default=os.getenv("S3_BUCKET"))
    parser.addoption('--s3-location', default=os.getenv("S3_LOCATION"))
    parser.addoption('--role', default=os.getenv("ROLE_NAME"))
    parser.addoption('--tag', default=os.getenv("IMAGE_TAG"))
    parser.addoption('--use-local-jobs', default=os.getenv("USE_LOCAL_JOBS", "True"))
    parser.addoption('--use-local-sim', default=os.getenv("USE_LOCAL_SIM", "True"))


@pytest.fixture(scope='session')
def region(request):
    return request.config.getoption('--region')


@pytest.fixture(scope='session')
def account(request):
    return request.config.getoption('--account')


@pytest.fixture(scope='session')
def role(request):
    return request.config.getoption('--role')


@pytest.fixture(scope='session')
def s3_bucket(request):
    return request.config.getoption('--s3-bucket')


@pytest.fixture(scope='session')
def s3_location(request):
    return request.config.getoption('--s3-location') or "image_test"


@pytest.fixture(scope='session')
def image_list(request, account, region):
    repository_name = request.config.getoption('--repository')
    image_tag = request.config.getoption('--tag')
    if repository_name and image_tag:
        return [f"{account}.dkr.ecr.{region}.amazonaws.com/{repository_name}:{image_tag}"]
    build_results_file = request.config.getoption('--from-build-results')
    if build_results_file:
        with open(build_results_file, "r") as build_file:
            build_results = json.load(build_file)
        return [image["ecr_url"] for image in build_results]
    raise Exception("No images specified for testing")


@pytest.fixture(scope='session')
def use_local_jobs(request):
    value = request.config.getoption('--use-local-jobs')
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value)


@pytest.fixture(scope='session')
def use_local_sim(request):
    value = request.config.getoption('--use-local-sim')
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value)


@pytest.fixture
def hyperparameters_json(pytester):
    pytester.makefile(
        ".json",
        hyperparameters="""
        {}
        """
    )
