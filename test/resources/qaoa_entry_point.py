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
import sys
import json
import time
import boto3

import numpy as np
import braket._sdk as braket_sdk

from braket.jobs import load_job_checkpoint, save_job_checkpoint, save_job_result
from braket.jobs.metrics import log_metric

import networkx as nx
import pennylane as qml
from matplotlib import pyplot as plt

from . import qaoa_utils


def init_pl_device(device_arn, num_nodes, max_parallel):
    return qml.device(
        "braket.aws.qubit",
        device_arn=device_arn,
        wires=num_nodes,
        shots=1000,
        s3_destination_folder=None,
        parallel=True,
        max_parallel=max_parallel,
    )


def start_function():
    hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
    with open(hp_file, "r") as f:
        hyperparams = json.load(f)
    print(hyperparams)

    num_nodes = int(hyperparams['num_nodes'])
    num_edges = int(hyperparams['num_edges'])
    p = int(hyperparams['p'])
    seed = int(hyperparams['seed'])
    max_parallel = int(hyperparams['max_parallel'])
    num_iterations = int(hyperparams['num_iterations'])
    interface = hyperparams['interface']

    # Import interface (PennyLane / TensorFlow / PyTorch)
    qaoa_utils.import_interface(interface=interface)

    g = nx.gnm_random_graph(num_nodes, num_edges, seed=seed)

    # Output figure to file
    output_dir = os.environ["AMZN_BRAKET_JOB_RESULTS_DIR"]
    positions = nx.spring_layout(g, seed=seed)
    nx.draw(g, with_labels=True, pos=positions, node_size=600)
    plt.savefig(f"{output_dir}/graph.png")

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
    dev = init_pl_device(device_arn, num_nodes, max_parallel)

    np.random.seed(seed)
    cost_function = qml.ExpvalCost(circuit, cost_h, dev, optimize=True, interface=interface)

    params = qaoa_utils.initialize_params(0.01 * np.random.uniform(size=[2, p]),
                                          interface=interface)

    optimizer = qaoa_utils.get_sgd_optimizer(params, interface=interface)
    print("Optimization start")

    for iteration in range(0, num_iterations):
        t0 = time.time()

        # Evaluates the cost, then does a gradient step to new params
        params, cost_before = qaoa_utils.get_cost_and_step(cost_function, params, optimizer,
                                                           interface=interface)
        # Convert params to a Numpy array so they're easier to handle for us
        np_params = qaoa_utils.convert_params_to_numpy(params, interface=interface)

        t1 = time.time()

        if iteration == 0:
            print("Initial cost:", cost_before)
        else:
            print(f"Cost at step {iteration}:", cost_before)

        # Log the current loss as a metric
        log_metric(
            metric_name="Cost",
            value=cost_before,
            iteration_number=iteration,
        )

        save_job_checkpoint(
            checkpoint_data={"iteration": iteration + 1, "params": np_params.tolist(),
                             "cost_before": cost_before},
            checkpoint_file_suffix="checkpoint-1",
        )

        with open(f"{output_dir}/cost_evolution.txt", "a") as f:
            f.write(f"{iteration} {cost_before} \n")

        print(f"Completed iteration {iteration + 1}")
        print(f"Time to complete iteration: {t1 - t0} seconds")

    final_cost = float(cost_function(params))

    with open(f"{output_dir}/cost_evolution.txt", "a") as f:
        f.write(f"{num_iterations} {final_cost} \n")
    print(f"Cost at step {num_iterations}:", final_cost)

    save_job_result(
        {
            "params": np_params.tolist(),
            "cost": final_cost
        }
    )

    print("Braket Container Run Success")


if __name__ == "__main__":
    start_function()
