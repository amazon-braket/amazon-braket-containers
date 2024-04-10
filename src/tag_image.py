import argparse
import json
import boto3
from botocore.exceptions import ClientError


def tag_image():
    parser = argparse.ArgumentParser(description="Program to tag built images")
    parser.add_argument("--build", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    with open(args.build, "r") as build_file:
        build_results = json.load(build_file)

    for image in build_results:
        _tag_image_with_value(image["ecr_url"], args.tag)


def _tag_image_with_value(ecr_url, tag_value):
    """Finds and tags an image with a given tag value. Unlike ECR, this function will not
    fail if the image already has the specified tag.

    Args:
        ecr_url: The URL of the image to tag.
        tag_value: The tag value to add to the image.
    """
    repo_path = ecr_url.rsplit('/', 1)[-1]
    repo_parts = repo_path.split(':')

    ecr = boto3.client('ecr')
    manifest = _call_ecr_to_get_manifest(ecr, repo_parts[0], repo_parts[1])

    # Tag it with the passed-in tag
    _call_ecr_to_tag_image(ecr, repo_parts[0], ecr_url, manifest, tag_value)


def _call_ecr_to_get_manifest(ecr, repository, tag):
    """
    Calls ECR to get the manifest for an image. The manifest is needed to update the image.

    Args:
        ecr: The ecr client
        repository: The ecr repository
        tag: The tag of the image to get the manifest for
    """
    get_image_response = ecr.batch_get_image(
        repositoryName=repository,
        imageIds=[
            {
                'imageTag': tag
            },
        ]
    )
    images = get_image_response["images"]

    num_images_found = len(images)
    if num_images_found != 1:
        raise Exception(f"Expected a single image, but {num_images_found} images"
                        f" found in {repository} with tag {tag}")

    image = images[0]
    return image["imageManifest"]


def _call_ecr_to_tag_image(ecr, repository, ecr_url, manifest, tag_value):
    """
    Calls ECR to tag an image, ignoring errors that arise when the image already has the tag.

    Args:
        ecr: The ecr client
        repository: The ecr repository
        ecr_url: The url of the image to tag
        manifest: The manifest of the image to tag
        tag_value: The tag value
    """
    try:
        print(f"Tagging {ecr_url} as {tag_value}")
        put_image_response = ecr.put_image(
            repositoryName=repository,
            imageManifest=manifest,
            imageTag=tag_value,
        )
        print(put_image_response)
    except ClientError as e:
        if e.response["Error"]["Code"] == "ImageAlreadyExistsException":
            print(f"Image {ecr_url} already tagged with {tag_value}.")
        else:
            raise e


if __name__ == "__main__":
    tag_image()