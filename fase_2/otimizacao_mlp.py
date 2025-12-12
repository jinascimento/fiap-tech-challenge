import random

def generate_individual():
    # geracao dos individuos
    # Os hiperparametros selecionados para otimização foram:

    # hidden_layer_sizes
    # Selecionado pela importância já que determina a arquitetura e capacidade da rede
    # Define o número de camadas ocultas e quantos neurônios

    # learning_rate_init
    # Controla a velocidade do aprendizado

    # activation
    # Define como os neurônios processam sinais e quão rápido o modelo ajusta os pesos

    # modelo que terá o individuo
    # {
    #   hidden_layer_sizes: (32, 16),
    #   activation: "relu",
    #   learning_rate_init: 0.01
    # }

    # ----- HIDDEN LAYER SIZES -----
    # Número de camadas escondidas: entre 1 e 3
    num_layers = random.randint(1, 3)

    # Neurônios por camada: entre 4 e 64
    hidden_layer_sizes = tuple(
        random.randint(4, 64) for _ in range(num_layers)
    )

    # ----- ACTIVATION FUNCTION -----
    activation_options = ["relu", "tanh", "logistic"]
    activation = random.choice(activation_options)

    # ----- LEARNING RATE INIT -----
    # Valor contínuo entre 0.0001 e 0.1
    learning_rate_init = 10 ** random.uniform(-4, -1)


    # ----- RETORNA O INDIVÍDUO -----
    return {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": activation,
        "learning_rate_init": learning_rate_init,
    }


def generate_population(size=10):
    return [generate_individual() for _ in range(size)]

if __name__ == "__main__":
    pop = generate_population(5)
    print(pop)
