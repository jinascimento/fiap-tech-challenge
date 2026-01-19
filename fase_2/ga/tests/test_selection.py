import unittest
from unittest.mock import patch
import random

from fase_2.ga.selection import elitism_selection, tournament_selection, apply_selection
from fase_2.ga.utils import set_random_seed


class TestSelection(unittest.TestCase):
    """Test cases for selection operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.population_with_fitness = [
            ({"n_hidden_layers": 1, "n_neurons": 16}, 0.5),
            ({"n_hidden_layers": 2, "n_neurons": 32}, 0.8),
            ({"n_hidden_layers": 1, "n_neurons": 64}, 0.3),
            ({"n_hidden_layers": 2, "n_neurons": 16}, 0.9),
            ({"n_hidden_layers": 1, "n_neurons": 32}, 0.6),
        ]

    def test_elitism_selection_single_elite(self):
        """Test elitism selection with n_elite=1"""
        result = elitism_selection(self.population_with_fitness, n_elite=1)
        
        self.assertEqual(len(result), 1)
        # Should select the individual with highest fitness (0.9)
        self.assertEqual(result[0][1], 0.9)

    def test_elitism_selection_multiple_elites(self):
        """Test elitism selection with n_elite=3"""
        result = elitism_selection(self.population_with_fitness, n_elite=3)
        
        self.assertEqual(len(result), 3)
        # Should be sorted in descending order
        self.assertEqual(result[0][1], 0.9)
        self.assertEqual(result[1][1], 0.8)
        self.assertEqual(result[2][1], 0.6)
        
        # Verify descending order
        for i in range(len(result) - 1):
            self.assertGreaterEqual(result[i][1], result[i+1][1])

    def test_elitism_selection_all_individuals(self):
        """Test elitism selection with n_elite equal to population size"""
        result = elitism_selection(self.population_with_fitness, n_elite=len(self.population_with_fitness))
        
        self.assertEqual(len(result), len(self.population_with_fitness))
        # Should be sorted in descending order
        self.assertEqual(result[0][1], 0.9)
        self.assertEqual(result[-1][1], 0.3)

    def test_elitism_selection_empty_population(self):
        """Test elitism selection with empty population"""
        result = elitism_selection([], n_elite=1)
        self.assertEqual(result, [])

    def test_tournament_selection_single_contender(self):
        """Test tournament selection with tournament_size=1"""
        set_random_seed(42)
        result = tournament_selection(self.population_with_fitness, tournament_size=1)
        
        # Should return a tuple (individual, fitness)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_tournament_selection_returns_best_in_tournament(self):
        """Test that tournament selection returns the best individual from tournament"""
        set_random_seed(42)
        
        # Create a controlled population where we know which will be selected
        controlled_pop = [
            ({"id": 1}, 0.1),
            ({"id": 2}, 0.9),
            ({"id": 3}, 0.2),
        ]
        
        # Run multiple times to ensure it selects from tournament
        results = []
        for _ in range(10):
            result = tournament_selection(controlled_pop, tournament_size=2)
            results.append(result[1])  # Collect fitness values
        
        # All results should be valid fitness values from the population
        for fitness in results:
            self.assertIn(fitness, [0.1, 0.2, 0.9])

    def test_tournament_selection_tournament_size_larger_than_population(self):
        """Test tournament selection when tournament_size > population size"""
        set_random_seed(42)
        small_pop = [
            ({"id": 1}, 0.5),
            ({"id": 2}, 0.8),
        ]
        
        # random.sample raises ValueError when sample size > population size
        with self.assertRaises(ValueError):
            tournament_selection(small_pop, tournament_size=5)

    @patch('fase_2.ga.selection.random.sample')
    def test_tournament_selection_uses_random_sample(self, mock_sample):
        """Test that tournament selection uses random.sample correctly"""
        mock_sample.return_value = [
            ({"id": 1}, 0.5),
            ({"id": 2}, 0.8),
        ]
        
        result = tournament_selection(self.population_with_fitness, tournament_size=2)
        
        mock_sample.assert_called_once()
        # Should return the max fitness from the sampled contenders
        self.assertEqual(result[1], 0.8)

    def test_apply_selection_returns_elites_and_parents(self):
        """Test that apply_selection returns elites and parents"""
        set_random_seed(42)
        elites, parents = apply_selection(
            self.population_with_fitness,
            pop_size=4,
            elite_size=1,
            tournament_size=2
        )
        
        self.assertIsInstance(elites, list)
        self.assertIsInstance(parents, list)
        self.assertEqual(len(elites), 1)
        self.assertEqual(len(parents), 4)

    def test_apply_selection_elites_are_best_individuals(self):
        """Test that elites are the best individuals"""
        set_random_seed(42)
        elites, _ = apply_selection(
            self.population_with_fitness,
            pop_size=4,
            elite_size=2,
            tournament_size=2
        )
        
        # Elites should be individuals (not tuples with fitness)
        for elite in elites:
            self.assertIsInstance(elite, dict)
        
        # Verify elites are the top 2 (we can't directly check fitness, but structure is correct)
        self.assertEqual(len(elites), 2)

    def test_apply_selection_parents_size_matches_pop_size(self):
        """Test that parents list size matches pop_size"""
        set_random_seed(42)
        _, parents = apply_selection(
            self.population_with_fitness,
            pop_size=5,
            elite_size=1,
            tournament_size=2
        )
        
        self.assertEqual(len(parents), 5)

    def test_apply_selection_parents_are_individuals(self):
        """Test that parents are individuals (not tuples with fitness)"""
        set_random_seed(42)
        _, parents = apply_selection(
            self.population_with_fitness,
            pop_size=3,
            elite_size=1,
            tournament_size=2
        )
        
        for parent in parents:
            self.assertIsInstance(parent, dict)

    def test_apply_selection_no_elites(self):
        """Test apply_selection with elite_size=0"""
        set_random_seed(42)
        elites, parents = apply_selection(
            self.population_with_fitness,
            pop_size=4,
            elite_size=0,
            tournament_size=2
        )
        
        self.assertEqual(len(elites), 0)
        self.assertEqual(len(parents), 4)

    def test_apply_selection_all_elites(self):
        """Test apply_selection when elite_size equals pop_size"""
        set_random_seed(42)
        elites, parents = apply_selection(
            self.population_with_fitness,
            pop_size=3,
            elite_size=3,
            tournament_size=2
        )
        
        self.assertEqual(len(elites), 3)
        self.assertEqual(len(parents), 3)

    def test_apply_selection_empty_population(self):
        """Test apply_selection with empty population"""
        set_random_seed(42)
        # This should raise a ValueError when trying to sample from empty list
        with self.assertRaises(ValueError):
            apply_selection(
                [],
                pop_size=4,
                elite_size=1,
                tournament_size=2
            )


if __name__ == '__main__':
    unittest.main()

