import random

def mutate(individual, mutation_rate=0.1, hyperparam_space = {}):
    mutated = individual.copy()

    for gene, values in hyperparam_space.items():
        if random.random() < mutation_rate:
            mutated[gene] = random.choice(values)

    return mutated


def apply_mutation(individuals, mutation_rate=0.1, hyperparam_space = {}):
    return [mutate(ind, mutation_rate, hyperparam_space) for ind in individuals]
