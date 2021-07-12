## My Project

TODO: Fill this README out!

Be sure to:

* Change the title in this README
* Edit your repository description on GitHub

### Getting started

This documentation uses the **base** container for provided examples.

Ensure you have [docker](https://docs.docker.com/get-docker/) client set-up on your system - osx/ec2

1. Clone the repo and set the following environment variables: 
    ```shell script
    export ACCOUNT_ID=<YOUR_ACCOUNT_ID>
    export REGION=us-west-2
    export REPOSITORY_NAME=beta-braket-base-containers
    ``` 

2. Login to ECR
    ```shell script
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com
    ``` 

3. Assuming your working directory is the cloned repo, create a virtual environment to use the repo and install requirements
    ```shell script
    pip install -r src/requirements.txt
    ``` 

### Building your image

1. To build all the dockerfiles specified in the buildspec.yml locally, use the command
    ```shell script
    python src/main.py --buildspec base/buildspec.yml --framework base
    ``` 
    The above step should take a while to complete the first time you run it since it will have to download all base layers 
    and create intermediate layers for the first time. 
    Subsequent runs should be much faster.



### Running tests locally
As part of your iteration with your PR, sometimes it is helpful to run your tests locally to avoid using too many
extraneous resources or waiting for a build to complete. The testing is supported using pytest. 

Similar to building locally, to test locally, you’ll need access to a personal/team AWS account. To test out:

1. Either on an EC2 instance with the amazon-braket-containers repo cloned, or on your local machine, make sure you have
the images you want to test locally (likely need to pull them from ECR)
   
2. Ensure you have AWS configured to be able to access the account you want to test in.
   
3. Make sure you set your AWS region:
    ```shell script
    export AWS_DEFAULT_REGION=us-west-2
    ```
   
4. Assuming your working directory is the cloned repo, create a virtual environment to use the repo and install requirements. 
   For example, for sagemaker tests:
    ```shell script
    pip install -r test/sagemaker_tests/base/jobs/requirements.txt
    ```
    
5. To run the tests, at minimum you'll need to specify the tag of the image you want to test, and the AWS role that should be
   used by tests
    ```shell script
    pytest --role Admin --tag 1.0-cpu-py37-ubuntu18.04-2021-06-18-20-09-42
    ```


### Structural Overview

This repo is structured to mirror the DLC GitHub repo for (deep-learning-containers) images. Until we can unify the
code bases, we need to copy much of the code to make the transition easier.  

We use the buildspec.yml and testspec.yml files as part of our CodeBuild pipeline to build these images
to our image repositories. 

The buildspec runs src/main.py, using environment variables to set the specifics of the build. As far as I can tell, 
environment variables are the easiest mechanism for doing this through CodeBuild. From main, we use utils.py to setup
the correct environment variables, and build it using image_builder.py and image.py. braket_container.py will
hold our custom code that we want to run when the image is invoked for a job.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

