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

import time
import boto3

import networkx as nx
from pennylane import numpy as np
import pennylane as qml

from braket.jobs import get_job_device_arn, save_job_checkpoint, save_job_result
from braket.jobs.metrics import log_metric

from . import qaoa_utils


def record_test_metrics(metric, start_time, device_arn, interface):
    cw_client = boto3.client("cloudwatch")
    cw_client.put_metric_data(
        MetricData=[{
            'MetricName': metric,
            'Dimensions': [
                {
                    'Name': 'TYPE',
                    'Value': 'braket_container_tests'
                },
                {
                    'Name': 'INTERFACE',
                    'Value': interface
                },
                {
                    'Name': 'DEVICE',
                    'Value': device_arn
                }
            ],
            'Unit': 'Seconds',
            'Value': time.time() - start_time
        }],
        Namespace='/aws/braket'
    )


def init_pl_device(device_arn, num_nodes, shots, max_parallel):
    if device_arn == "local:none/none":
        return qml.device(
            "braket.local.qubit",
            wires=num_nodes,
            shots=shots
        )

    return qml.device(
        "braket.aws.qubit",
        device_arn=device_arn,
        wires=num_nodes,
        shots=shots,
        s3_destination_folder=None,
        parallel=True,
        max_parallel=max_parallel,
    )


def entry_point(
    p: int,
    seed: int,
    max_parallel: int,
    num_iterations: int,
    stepsize: float,
    shots: int,
    pl_interface: str,
    start_time: float,
):
    device_arn = get_job_device_arn()
    record_test_metrics('Startup', start_time, device_arn, pl_interface)

    interface = qaoa_utils.QAOAInterface.get_interface(pl_interface)

    # Get PennyLane-compatible interface name
    if hasattr(interface, 'get_pennylane_interface'):
        pennylane_interface = interface.get_pennylane_interface()
    else:
        pennylane_interface = pl_interface

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

    dev = init_pl_device(device_arn, num_nodes, shots, max_parallel)

    np.random.seed(seed)

    @qml.qnode(dev, interface=pennylane_interface)
    def cost_function(params):
        circuit(params)
        return qml.expval(cost_h)

    params = interface.initialize_params(0.01 * np.random.uniform(size=[2, p]))

    optimizer = interface.get_sgd_optimizer(stepsize, params)
    print("Optimization start")

    if interface.supports_full_optimization():
        # CUDA-Q path: run full optimization
        t0 = time.time()
        final_params, final_cost = interface.run_full_optimization(
            cost_function, params, optimizer, num_iterations)
        t1 = time.time()
        
        print(f"Initial cost: {float(cost_function(params))}")
        print(f"Final cost: {final_cost}")
        
        # Simulate step-by-step logging for consistency
        for iteration in range(num_iterations + 1):
            cost_value = final_cost if iteration == num_iterations else float(cost_function(params))
            log_metric(
                metric_name="Cost",
                value=cost_value,
                iteration_number=iteration,
            )
        
        np_params = interface.convert_params_to_numpy(final_params)
        save_job_checkpoint(
            checkpoint_data={
                "iteration": num_iterations,
                "params": np_params.tolist(),
                "cost_before": final_cost,
            },
            checkpoint_file_suffix="checkpoint-1",
        )
        
        print(f"Completed full optimization in {t1 - t0} seconds")
        params = final_params
        
    else:
        # Traditional path: step-by-step optimization
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
    np_params = interface.convert_params_to_numpy(params)

    save_job_result({"params": np_params.tolist(), "cost": final_cost})

    record_test_metrics('Total', start_time, device_arn, pl_interface)
    print("Braket Container Run Success")
