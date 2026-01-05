from fase_2.ga.generate_individual import generate_individual


def generate_population(size=5, hyperparam_space = {}):
    return [generate_individual(hyperparam_space) for _ in range(size)]

