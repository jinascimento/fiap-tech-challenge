import random

def generate_individual(hyperparam_space):
    return {
        "n_hidden_layers": random.choice(hyperparam_space["n_hidden_layers"]),
        "n_neurons": random.choice(hyperparam_space["n_neurons"]),
        "learning_rate_init": random.choice(hyperparam_space["learning_rate_init"]),
        "activation": random.choice(hyperparam_space["activation"]),
    }
