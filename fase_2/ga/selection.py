import random

def elitism_selection(population, n_elite=1):
    sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)
    return sorted_pop[:n_elite]


def tournament_selection(population, tournament_size=2):
    contenders = random.sample(population, tournament_size)
    return max(contenders, key=lambda x: x[1])


def apply_selection(population, pop_size=4, elite_size=1, tournament_size=2):
    # Seleciona o individuo numero 1
    elites = elitism_selection(population, elite_size)
    elites = [ind for ind, _ in elites]

    parents = []
    while len(parents) < pop_size:
        winner, _ = tournament_selection(population, tournament_size)
        parents.append(winner)

    return elites, parents
