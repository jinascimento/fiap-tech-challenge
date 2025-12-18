# definindo os limites para mutacao e geracao dos individuos iniciais
HYPERPARAM_SPACE = {
    "n_hidden_layers": [1, 2],
    "n_neurons": [16, 32, 64],
    "learning_rate_init": [0.001, 0.01, 0.1],
    "activation": ["relu", "tanh"]
}