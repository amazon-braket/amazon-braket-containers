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

import argparse
import os
import utils
from image_builder import image_builder


def main():
    parser = argparse.ArgumentParser(description="Program to build docker images")
    parser.add_argument("--buildspec", type=str)
    parser.add_argument("--framework", type=str, required=True)
    args = parser.parse_args()

    utils.build_setup(args.framework)

    if args.buildspec is None and args.framework is not None:
        args.buildspec = os.path.join(args.framework, "buildspec.yml")

    image_builder(args.buildspec)


if __name__ == "__main__":
    main()