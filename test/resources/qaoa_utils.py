from abc import ABC, abstractmethod


class QAOAInterface(ABC):
    @classmethod
    @abstractmethod
    # Initialize params to the appropriate format for the autodiff library.
    def initialize_params(cls, np_array):
        pass

    @classmethod
    @abstractmethod
    # Returns the gradient descent optimizer to use, based on the ML framework.
    def get_sgd_optimizer(cls, stepsize, params):
        pass

    @classmethod
    @abstractmethod
    # Convert params to base Numpy arrays.
    def convert_params_to_numpy(cls, params):
        pass

    @classmethod
    @abstractmethod
    # Evaluate the cost function, then take a step.
    def get_cost_and_step(cls, cost_function, params, optimizer):
        pass

    @classmethod
    def supports_full_optimization(cls):
        """Return True if this interface handles full optimization internally"""
        return False
    
    @classmethod
    def run_full_optimization(cls, cost_function, initial_params, optimizer, num_iterations):
        """Run complete optimization (only for interfaces that support it)"""
        raise NotImplementedError("This interface doesn't support full optimization")

    @staticmethod
    # Get the appropriate interface to use based on the string input.
    def get_interface(interface):
        if interface == "autograd":
            return AutogradInterface()
        elif interface == "tf":
            return TensorFlowInterface()
        elif interface == "torch":
            return PyTorchInterface()
        elif interface == "cudaq":
            return CudaQInterface()
        else:
            raise ValueError(f"Interface {interface} is invalid.")


class AutogradInterface(QAOAInterface):
    # Import only in __init__ so we don't import a library that doesn't exist in the container.
    def __init__(self):
        global qml
        qml = __import__("pennylane", globals(), locals())
        # Equiv to: import pennylane as qml

    @classmethod
    def initialize_params(cls, np_array):
        return np_array

    @classmethod
    def get_sgd_optimizer(cls, stepsize, params):
        return qml.GradientDescentOptimizer(stepsize=stepsize)

    @classmethod
    def convert_params_to_numpy(cls, params):
        return params.numpy()

    @classmethod
    def get_cost_and_step(cls, cost_function, params, optimizer):
        params, cost_before = optimizer.step_and_cost(cost_function, params)
        return params, float(cost_before)


class TensorFlowInterface(QAOAInterface):
    # Import only in __init__ so we don't import a library that doesn't exist in the container.
    def __init__(self):
        global tf
        tf = __import__("tensorflow", globals(), locals())
        # Equiv to: import tensorflow as tf

    @classmethod
    def initialize_params(cls, np_array):
        return tf.Variable(np_array, dtype=tf.float64)

    @classmethod
    def get_sgd_optimizer(cls, stepsize, params):
        return tf.keras.optimizers.legacy.SGD(learning_rate=stepsize)

    @classmethod
    def convert_params_to_numpy(cls, params):
        return params.numpy()

    @classmethod
    def get_cost_and_step(cls, cost_function, params, optimizer):
        def tf_cost():
            global _cached_cost_before
            _cached_cost_before = cost_function(params)
            return _cached_cost_before

        optimizer.minimize(tf_cost, params)
        cost_before = _cached_cost_before

        # Alternative:
        # with tf.GradientTape() as tape:
        #     cost_before = cost_function(params)
        #
        # gradients = tape.gradient(cost_before, params)
        # optimizer.apply_gradients(((gradients, params),))

        return params, float(cost_before)


class PyTorchInterface(QAOAInterface):
    # Import only in __init__ so we don't import a library that doesn't exist in the container.
    def __init__(self):
        global torch
        torch = __import__("torch", globals(), locals())
        # Equiv to: import torch

    @classmethod
    def initialize_params(cls, np_array):
        return torch.tensor(np_array, requires_grad=True)

    @classmethod
    def get_sgd_optimizer(cls, stepsize, params):
        return torch.optim.SGD([params], lr=stepsize)

    @classmethod
    def convert_params_to_numpy(cls, params):
        return params.detach().numpy()

    @classmethod
    def get_cost_and_step(cls, cost_function, params, optimizer):
        optimizer.zero_grad()
        cost_before = cost_function(params)
        cost_before.backward()
        optimizer.step()
        return params, float(cost_before)


class CudaQInterface(QAOAInterface):
    def __init__(self):
        global cudaq, np
        cudaq = __import__("cudaq", globals(), locals())
        np = __import__("numpy", globals(), locals())

    @classmethod
    def initialize_params(cls, np_array):
        return np_array

    @classmethod
    def supports_full_optimization(cls):
        return True

    @classmethod
    def get_sgd_optimizer(cls, stepsize, params):
        optimizer = cudaq.optimizers.Adam()
        gradient = cudaq.gradients.CentralDifference()
        return {"optimizer": optimizer, "gradient": gradient}

    @classmethod
    def convert_params_to_numpy(cls, params):
        return np.array(params)

    @classmethod
    def get_cost_and_step(cls, cost_function, params, optimizer):
        # Not used for CUDA-Q (uses full optimization instead)
        cost_before = cost_function(params)
        return params, float(cost_before)
    
    @classmethod
    def run_full_optimization(cls, cost_function, initial_params, optimizer_config, num_iterations):
        optimizer = optimizer_config["optimizer"]
        gradient = optimizer_config["gradient"]
        
        def objective_function(parameter_vector):
            params_shaped = np.array(parameter_vector).reshape(initial_params.shape)
            cost = cost_function(params_shaped)
            
            def cost_func_flat(x):
                return cost_function(np.array(x).reshape(initial_params.shape))
            
            gradient_vector = gradient.compute(parameter_vector, cost_func_flat, cost)
            return cost, gradient_vector
        
        flat_params = initial_params.flatten()
        final_cost, final_params = optimizer.optimize(dimensions=len(flat_params), 
                                                     function=objective_function)
        
        return np.array(final_params).reshape(initial_params.shape), final_cost
    
    @classmethod
    def get_pennylane_interface(cls):
        return "autograd"



