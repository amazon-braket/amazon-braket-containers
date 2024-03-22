### Getting started

[![Python package](https://github.com/aws/amazon-braket-containers/actions/workflows/python-package.yml/badge.svg)](https://github.com/aws/amazon-braket-containers/actions/workflows/python-package.yml)

This documentation uses the **base** container for provided examples.

Ensure you have [docker](https://docs.docker.com/get-docker/) client set-up on your system - osx/ec2

Create an ECR repository in your AWS account. In this example, we'll assume it's called "amazon-braket-base-jobs"

1. Clone the repo and set the following environment variables:
    ```shell script
    export REGION=us-west-2
    export REPOSITORY_NAME=amazon-braket-base-jobs
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    ```
   Make sure you set the repository name to the name you created in your AWS account.

2. Login to ECR
    ```shell script
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ``` 

3. Assuming your working directory is the cloned repo, create a virtual environment to use the repo and install
   requirements
    ```shell script
    pip install -r src/requirements.txt
    ``` 

### Building your image

1. To build all the dockerfiles specified in the buildspec.yml locally, use the command
    ```shell script
    python src/main.py --framework base
    ``` 
   The above step should take a while to complete the first time you run it since it will have to download all base
   layers and create intermediate layers for the first time. Subsequent runs should be much faster.

   To build other frameworks change the framework flag to one of the other supported frameworks: {pytorch, tensorflow}.

### Running tests locally

As part of your iteration with your PR, sometimes it is helpful to run your tests locally to avoid using too many
extraneous resources or waiting for a build to complete. The testing is supported using pytest.

Similar to building locally, to test locally, you’ll need access to a personal/team AWS account. To test out:

1. Ensure you have AWS configured to be able to access the account you want to test in.

2. Make sure you set your AWS region:
    ```shell script
    export AWS_DEFAULT_REGION=us-west-2
    ```

3. Assuming your working directory is the cloned repo, create a virtual environment to use the repo and install
   requirements.
    ```shell script
    pip install -r test/requirements.txt
    ```

4. To run the unit tests:
    ```shell script
    pytest test/unit_tests
    ```    

5. To run the SageMaker integration tests, at minimum you'll need to specify the tag of the image you want to test, the
   AWS role that should be used by tests, and the S3 location where a test file can be uploaded. Create this bucket in
   S3 before you run the test.
    ```shell script
    pytest test/sagemaker_tests --role Admin --tag latest --s3-bucket amazon-braket-123456
    ```

6. To run the Braket integration tests, at minimum you'll need to specify the tag of the image you want to test, the AWS
   role that should be used by tests. The framework to test should be included in the test path.
    ```shell script
    pytest test/braket_tests/base --role service-role/AmazonBraketJobsExecutionRole --tag latest
    ```

### Structural Overview

This repo is structured to mirror the DLC GitHub repo for (deep-learning-containers) images. Until we can unify the code
bases, we need to copy much of the code to make the transition easier.

We use the buildspec.yml and testspec.yml files as part of our CodeBuild pipeline to build these images to our image
repositories.

The buildspec runs src/main.py, using environment variables to set the specifics of the build. As far as I can tell,
environment variables are the easiest mechanism for doing this through CodeBuild. From main, we use utils.py to setup
the correct environment variables, and build it using image_builder.py and image.py. braket_container.py will hold our
custom code that we want to run when the image is invoked for a job.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Support

- **Source Code:** https://github.com/aws/amazon-braket-containers
- **Issue Tracker:** https://github.com/aws/amazon-braket-containers/issues
- **General Questions:** https://quantumcomputing.stackexchange.com/questions/ask?Tags=amazon-braket

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

## License

This project is licensed under the Apache-2.0 License.

EOF