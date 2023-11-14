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

import json
import os
import time
import boto3

import networkx as nx
from pennylane import numpy as np
import pennylane as qml

from braket.jobs import save_job_checkpoint, save_job_result
from braket.jobs.metrics import log_metric

from . import qaoa_utils


def record_test_metrics(metric, start_time, interface):
    cw_client = boto3.client("cloudwatch")
    cw_client.put_metric_data(
        MetricData=[{
            'MetricName': metric,
            'Dimensions': [
                {
                    'Name': 'TYPE',
                    'Value': 'braket_tests'
                },
                {
                    'Name': 'INTERFACE',
                    'Value': interface
                }
            ],
            'Unit': 'Seconds',
            'Value': time.time() - start_time
        }],
        Namespace='/aws/braket'
    )


def init_pl_device(device_arn, num_nodes, shots, max_parallel):
    return qml.device(
        "braket.aws.qubit",
        device_arn=device_arn,
        wires=num_nodes,
        shots=shots,
        s3_destination_folder=None,
        parallel=True,
        max_parallel=max_parallel,
    )


def start_function():
    # Read the hyperparameters
    hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
    with open(hp_file, "r") as f:
        hyperparams = json.load(f)
    print(hyperparams)

    p = int(hyperparams["p"])
    seed = int(hyperparams["seed"])
    max_parallel = int(hyperparams["max_parallel"])
    num_iterations = int(hyperparams["num_iterations"])
    stepsize = float(hyperparams["stepsize"])
    shots = int(hyperparams["shots"])
    pl_interface = hyperparams["interface"]
    start_time = float(hyperparams["start_time"])

    record_test_metrics('Startup', start_time, pl_interface)

    interface = qaoa_utils.QAOAInterface.get_interface(pl_interface)

    g = nx.gnm_random_graph(4, 4, seed=seed)
    num_nodes = len(g.nodes)

    # Set up the QAOA problem
    cost_h, mixer_h = qml.qaoa.maxcut(g)

    def qaoa_layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)

    def circuit(params, **kwargs):
        for i in range(num_nodes):
            qml.Hadamard(wires=i)
        qml.layer(qaoa_layer, p, params[0], params[1])

    device_arn = os.environ["AMZN_BRAKET_DEVICE_ARN"]
    dev = init_pl_device(device_arn, num_nodes, shots, max_parallel)

    np.random.seed(seed)

    @qml.qnode(dev, interface=pl_interface)
    def cost_function(params):
        circuit(params)
        return qml.expval(cost_h)

    params = interface.initialize_params(0.01 * np.random.uniform(size=[2, p]))

    optimizer = interface.get_sgd_optimizer(stepsize, params)
    print("Optimization start")

    for iteration in range(num_iterations):
        t0 = time.time()

        # Evaluates the cost, then does a gradient step to new params
        params, cost_before = interface.get_cost_and_step(cost_function, params, optimizer)
        # Convert params to a Numpy array so they're easier to handle for us
        np_params = interface.convert_params_to_numpy(params)

        t1 = time.time()

        if iteration == 0:
            print("Initial cost:", cost_before)
        else:
            print(f"Cost at step {iteration}:", cost_before)

        # Log the loss before the update step as a metric
        log_metric(
            metric_name="Cost",
            value=cost_before,
            iteration_number=iteration,
        )

        # Save the current params and previous cost to a checkpoint
        save_job_checkpoint(
            checkpoint_data={
                "iteration": iteration + 1,
                "params": np_params.tolist(),
                "cost_before": cost_before,
            },
            checkpoint_file_suffix="checkpoint-1",
        )

        print(f"Completed iteration {iteration + 1}")
        print(f"Time to complete iteration: {t1 - t0} seconds")

    final_cost = float(cost_function(params))
    log_metric(
        metric_name="Cost",
        value=final_cost,
        iteration_number=num_iterations,
    )

    print(f"Cost at step {num_iterations}:", final_cost)

    save_job_result({"params": np_params.tolist(), "cost": final_cost})

    record_test_metrics('Total', start_time, pl_interface)
    print("Braket Container Run Success")


if __name__ == "__main__":
    start_function()
