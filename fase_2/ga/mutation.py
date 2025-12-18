import random
from fase_2.ga.config import HYPERPARAM_SPACE

def mutate(individual, mutation_rate=0.1):
    mutated = individual.copy()

    for gene, values in HYPERPARAM_SPACE.items():
        if random.random() < mutation_rate:
            mutated[gene] = random.choice(values)

    return mutated


def apply_mutation(individuals, mutation_rate=0.1):
    return [mutate(ind, mutation_rate) for ind in individuals]
