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

import boto3
import constants


class Metrics(object):
    def __init__(self, context="DEV", region="us-west-2", namespace="braket-container-metrics"):
        """
        Constructor for the metrics object.

        Parameters:
            context: the value of the BuildContext dimension
            region: the AWS region
            namespace: the metrics namespace
        """
        self.client = boto3.Session(region_name=region).client("cloudwatch")
        self.context = context
        self.namespace = namespace

    def push(self, name, unit, value, metrics_info):
        """
        Pushes metrics to CloudWatch Metrics.

        Parameters:
            name: the name of the metric
            unit: the metric unit
            value: the metric value
            metrics_info: additional dimensions for the metric
        """

        dimensions = [{"Name": "BuildContext", "Value": self.context}]

        for key in metrics_info:
            dimensions.append({"Name": key, "Value": metrics_info[key]})

        try:
            response = self.client.put_metric_data(
                MetricData=[
                    {
                        "MetricName": name,
                        "Dimensions": dimensions,
                        "Unit": unit,
                        "Value": value,
                    },
                ],
                Namespace=self.namespace,
            )
        except Exception as e:
            raise Exception(str(e))

        return response

    def push_image_metrics(self, image):
        """
        Pushes metrics about a docker image to CloudWatch Metrics.

        Parameters:
            image: docker image information.
        """
        info = {
            "framework": image.framework,
            "version": image.version,
            "device_type": image.device_type,
            "python_version": image.python_version,
            "image_type": image.image_type,
        }
        if image.build_status == constants.NOT_BUILT:
            return None
        build_time = (image.summary["end_time"] - image.summary["start_time"]).seconds
        build_status = image.build_status

        self.push("build_time", "Seconds", build_time, info)
        self.push("build_status", "None", build_status, info)

        if image.build_status == constants.SUCCESS:
            image_size = image.summary["image_size"]
            self.push("image_size", "Bytes", image_size, info)
