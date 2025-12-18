from fase_2.ga.generate_individual import generate_individual


def generate_population(size=5):
    return [generate_individual() for _ in range(size)]

