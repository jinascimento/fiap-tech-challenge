from utils.module_diabetes import load_dataset, prepare_dataset, train_mlp

from fase_2.ga.generate_population import generate_population
from fase_2.ga.fitness import calculate_fitness
from fase_2.ga.selection import apply_selection
from fase_2.ga.crossover import apply_crossover
from fase_2.ga.mutation import apply_mutation
from fase_2.ga.utils import set_random_seed

def run_ga(config):
    set_random_seed(config["random_seed"])
    df = load_dataset()
    df_processed = prepare_dataset(df)

    population = generate_population(config["population_size"])
    population_with_fitness = []

    for individual in population:
        # Conversão cromossomo → MLP
        hidden_layer_sizes = tuple([individual["n_neurons"]] * individual["n_hidden_layers"])

        model, X_val, y_val = train_mlp(df_processed,
            {
                "hidden_layer_sizes": hidden_layer_sizes,
                "activation": individual["activation"],
                "learning_rate_init": individual["learning_rate_init"],
            }
        )

        fitness = calculate_fitness(model, X_val, y_val)
        population_with_fitness.append((individual, fitness))

    elites, selected_population = apply_selection(population_with_fitness, config["population_size"], config["elite_size"], config["tournament_size"])
    offspring = apply_crossover(selected_population, config["population_size"], config["crossover_rate"])
    offspring = apply_mutation(offspring, config["mutation_rate"])

    new_population = elites + offspring

    print("\n=== Nova população ===")
    for ind in new_population:
        print(ind)

if __name__ == "__main__":
    config = {
        "population_size": 5,
        "elite_size": 1,
        "tournament_size": 2,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "random_seed": 42
    }

    run_ga(config)
