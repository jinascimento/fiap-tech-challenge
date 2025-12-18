import random
from fase_2.ga.config import HYPERPARAM_SPACE

def generate_individual():
    return {
        "n_hidden_layers": random.choice(HYPERPARAM_SPACE["n_hidden_layers"]),
        "n_neurons": random.choice(HYPERPARAM_SPACE["n_neurons"]),
        "learning_rate_init": random.choice(HYPERPARAM_SPACE["learning_rate_init"]),
        "activation": random.choice(HYPERPARAM_SPACE["activation"]),
    }
