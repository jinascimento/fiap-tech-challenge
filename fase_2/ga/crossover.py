import random

def crossover(parent1, parent2):
    return {
        gene: random.choice([parent1[gene], parent2[gene]])
        for gene in parent1.keys()
    }


def apply_crossover(parents, pop_size, elite_size, crossover_rate=0.8):
    new_individuals = []
    individuals_size = pop_size - elite_size

    while len(new_individuals) < individuals_size:
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)

        if random.random() < crossover_rate:
            child = crossover(parent1, parent2)
        else:
            child = parent1.copy()

        new_individuals.append(child)

    return new_individuals
