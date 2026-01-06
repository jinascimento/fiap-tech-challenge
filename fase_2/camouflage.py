import random

import pygame

# Constants
# ----------------------------------------------------------------------------
WIDTH, HEIGHT = 800, 600
POPULATION_SIZE = 100
INDIVIDUAL_SIZE = 20
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 10
FPS = 60
BACKGROUND_COLOR = (50, 150, 50)

# UI Settings
# ----------------------------------------------------------------------------
PLOT_WIDTH = 200
PLOT_HEIGHT = 100
INFO_WIDTH = 220
INFO_HEIGHT = 90


# UI Functions
# ----------------------------------------------------------------------------
def draw_ui_backgrounds(screen, info_bg, plot_bg):
    screen.blit(info_bg, (10, 10))
    screen.blit(plot_bg, (WIDTH - PLOT_WIDTH - 20, HEIGHT - PLOT_HEIGHT - 20))


def draw_text_info(screen, font, generation, population):
    color = (220, 220, 220)
    best_fit = population[0].fitness

    lines = [
        f"Generation: {generation}",
        f"Best Error: {best_fit} (0 is perfect)",
        "SPACE: New Environment",
        "K: Quit",
    ]

    for i, line in enumerate(lines):
        text = font.render(line, True, color)
        screen.blit(text, (15, 15 + (i * 20)))


def draw_plot(screen, history, x_pos, y_pos, width, height):
    if len(history) < 2:
        return

    max_val = max(history) if max(history) > 0 else 1
    points = []

    # Border
    pygame.draw.rect(screen, (255, 255, 255), (x_pos, y_pos, width, height), 1)

    for i, val in enumerate(history):
        x = x_pos + (i * (width / len(history)))

        norm_h = (val / max_val) * (height - 2)

        y = (y_pos + height) - norm_h
        points.append((x, y))

    pygame.draw.lines(screen, (255, 50, 50), False, points, 2)


# Helpers
# ----------------------------------------------------------------------------
def crossover(father1, father2):
    new_dna = []
    for i in range(3):
        gene = father1.dna[i] if random.random() > 0.5 else father2.dna[i]
        new_dna.append(gene)
    return Individual(new_dna)


def mutation(individual):
    for i in range(3):
        if random.random() < 0.3:
            change = random.randint(-MUTATION_STRENGTH, MUTATION_STRENGTH)
            individual.dna[i] = max(0, min(255, individual.dna[i] + change))

        if random.random() < MUTATION_RATE:
            individual.dna[i] = random.randint(0, 255)


def tournament(population, k=3):
    competitors = random.sample(population, k)
    return min(competitors, key=lambda x: x.fitness)


# Core
# ----------------------------------------------------------------------------
class Individual:
    def __init__(self, dna=None):
        if not dna:
            self.dna = [random.randint(0, 255) for _ in range(3)]
        else:
            self.dna = dna

        self.x = random.randint(0, WIDTH - INDIVIDUAL_SIZE)
        self.y = random.randint(0, HEIGHT - INDIVIDUAL_SIZE)
        self.fitness = 0

    def draw(self, screen):
        rect = (self.x, self.y, INDIVIDUAL_SIZE, INDIVIDUAL_SIZE)
        pygame.draw.rect(screen, self.dna, rect)

        # border
        pygame.draw.rect(screen, (50, 50, 50), rect, 1)

    def calculate_fitness(self, target):
        diff_r = abs(self.dna[0] - target[0])
        diff_g = abs(self.dna[1] - target[1])
        diff_b = abs(self.dna[2] - target[2])
        self.fitness = diff_r + diff_g + diff_b


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Genetic Algorithm: Camouflage Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)

    # Pre-render semi-transparent backgrounds used for data information
    info_bg = pygame.Surface((INFO_WIDTH, INFO_HEIGHT))
    info_bg.set_alpha(180)
    info_bg.fill((0, 0, 0))

    plot_bg = pygame.Surface((PLOT_WIDTH, PLOT_HEIGHT))
    plot_bg.set_alpha(180)
    plot_bg.fill((0, 0, 0))

    # Initial State
    population = [Individual() for _ in range(POPULATION_SIZE)]
    generation = 1
    current_bg_color = list(BACKGROUND_COLOR)

    error_history = []

    running = True

    while running:
        clock.tick(FPS)  # Controls loop speed smoothly

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    current_bg_color = [random.randint(0, 255) for _ in range(3)]
                if event.key == pygame.K_k:
                    running = False

        for ind in population:
            ind.calculate_fitness(current_bg_color)

        # Metrics
        avg_error = sum(ind.fitness for ind in population) / POPULATION_SIZE
        error_history.append(avg_error)
        if len(error_history) > PLOT_WIDTH:
            error_history.pop(0)

        # Sort the best (lowest error) first
        population.sort(key=lambda x: x.fitness)

        # Reproduction
        new_pop = []

        # 1. Elitism (fixed: 2% ensures keeping the absolute best pair)
        elite_count = int(POPULATION_SIZE * 0.02)
        new_pop.extend(population[:elite_count])

        # 2. Immigrants (exploration)
        immigrant_count = int(POPULATION_SIZE * 0.10)
        for _ in range(immigrant_count):
            new_pop.append(Individual())

        # 3. Crossover (tournament)
        while len(new_pop) < POPULATION_SIZE:
            p1 = tournament(population)
            p2 = tournament(population)
            child = crossover(p1, p2)
            mutation(child)
            new_pop.append(child)

        population = new_pop
        generation += 1

        screen.fill(current_bg_color)

        # Draw individuals reversed (worst first, best on top)
        # This makes the "survivors" visible on top of the mess
        for ind in reversed(population):
            ind.draw(screen)

        # UI Layers
        draw_ui_backgrounds(screen, info_bg, plot_bg)
        draw_text_info(screen, font, generation, population)
        draw_plot(
            screen,
            error_history,
            WIDTH - PLOT_WIDTH - 20,
            HEIGHT - PLOT_HEIGHT - 20,
            PLOT_WIDTH,
            PLOT_HEIGHT,
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
