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


def import_interface(interface='autograd'):
    if interface == 'autograd':
        global qml
        qml = __import__('pennylane', globals(), locals())
    elif interface == 'tf':
        global tf
        tf = __import__('tensorflow', globals(), locals())
    elif interface == 'torch':
        global torch
        torch = __import__('torch', globals(), locals())


def initialize_params(np_array, interface='autograd'):
    if interface == 'autograd':
        return np_array
    elif interface == 'tf':
        return tf.Variable(np_array, dtype=tf.float64)
    elif interface == 'torch':
        return torch.tensor(np_array, requires_grad=True)
    else:
        pass


def get_sgd_optimizer(params, interface='autograd', stepsize=0.1):
    if interface == 'autograd':
        return qml.GradientDescentOptimizer(stepsize=stepsize)
    elif interface == 'tf':
        return tf.keras.optimizers.SGD(learning_rate=stepsize)
    elif interface == 'torch':
        return torch.optim.SGD([params], lr=stepsize)
    else:
        pass


def convert_params_to_numpy(params, interface='autograd'):
    if interface == 'autograd':
        return params.numpy()
    elif interface == 'tf':
        return params.numpy()
    elif interface == 'torch':
        return params.detach().numpy()


def get_cost_and_step(cost_function, params, optimizer, interface='autograd'):
    if interface == 'autograd':
        params, cost_before = optimizer.step_and_cost(cost_function, params)
    elif interface == 'tf':
        with tf.GradientTape() as tape:
            cost_before = cost_function(params)
        gradients = tape.gradient(cost_before, params)
        optimizer.apply_gradients(((gradients, params),))
    elif interface == 'torch':
        optimizer.zero_grad()
        cost_before = cost_function(params)
        cost_before.backward()
        optimizer.step()
    else:
        pass

    return params, float(cost_before)
