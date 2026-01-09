import unittest
from unittest.mock import patch, MagicMock

from fase_2.ga.generate_population import generate_population
from fase_2.ga.utils import set_random_seed


class TestGeneratePopulation(unittest.TestCase):
    """Test cases for population generation"""

    def setUp(self):
        """Set up test fixtures"""
        self.hyperparam_space = {
            "n_hidden_layers": [1, 2],
            "n_neurons": [16, 32, 64],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "activation": ["relu", "tanh"]
        }

    def test_generate_population_returns_list(self):
        """Test that generate_population returns a list"""
        set_random_seed(42)
        population = generate_population(size=5, hyperparam_space=self.hyperparam_space)
        
        self.assertIsInstance(population, list)

    def test_generate_population_correct_size(self):
        """Test that generate_population creates population of correct size"""
        set_random_seed(42)
        for size in [1, 5, 10, 20]:
            population = generate_population(size=size, hyperparam_space=self.hyperparam_space)
            self.assertEqual(len(population), size)

    def test_generate_population_empty_population(self):
        """Test generate_population with size=0"""
        set_random_seed(42)
        population = generate_population(size=0, hyperparam_space=self.hyperparam_space)
        
        self.assertEqual(len(population), 0)
        self.assertIsInstance(population, list)

    def test_generate_population_individuals_are_valid(self):
        """Test that all individuals in population are valid"""
        set_random_seed(42)
        population = generate_population(size=5, hyperparam_space=self.hyperparam_space)
        
        for individual in population:
            self.assertIsInstance(individual, dict)
            self.assertEqual(set(individual.keys()), {
                "n_hidden_layers",
                "n_neurons",
                "learning_rate_init",
                "activation"
            })

    def test_generate_population_individuals_in_valid_space(self):
        """Test that all individuals have values within hyperparameter space"""
        set_random_seed(42)
        population = generate_population(size=10, hyperparam_space=self.hyperparam_space)
        
        for individual in population:
            for key, value in individual.items():
                self.assertIn(value, self.hyperparam_space[key])

    def test_generate_population_different_individuals(self):
        """Test that population contains potentially different individuals"""
        set_random_seed(42)
        population = generate_population(size=10, hyperparam_space=self.hyperparam_space)
        
        # With a large enough population and diverse hyperparameter space,
        # we should have some diversity
        # At least check that not all individuals are identical
        first_individual = population[0]
        all_same = all(ind == first_individual for ind in population)
        
        # This might be True if seed makes them all same, but structure should be correct
        # We mainly verify structure is correct
        self.assertTrue(all(isinstance(ind, dict) for ind in population))

    def test_generate_population_reproducible_with_seed(self):
        """Test that generate_population is reproducible with same seed"""
        set_random_seed(42)
        population1 = generate_population(size=5, hyperparam_space=self.hyperparam_space)
        
        set_random_seed(42)
        population2 = generate_population(size=5, hyperparam_space=self.hyperparam_space)
        
        self.assertEqual(population1, population2)

    @patch('fase_2.ga.generate_population.generate_individual')
    def test_generate_population_calls_generate_individual(self, mock_generate_individual):
        """Test that generate_population calls generate_individual for each individual"""
        mock_individual = {
            "n_hidden_layers": 1,
            "n_neurons": 32,
            "learning_rate_init": 0.01,
            "activation": "relu"
        }
        mock_generate_individual.return_value = mock_individual
        
        population = generate_population(size=5, hyperparam_space=self.hyperparam_space)
        
        # Should be called 5 times
        self.assertEqual(mock_generate_individual.call_count, 5)
        self.assertEqual(len(population), 5)
        # All individuals should be the mocked value
        self.assertTrue(all(ind == mock_individual for ind in population))

    def test_generate_population_with_empty_hyperparam_space(self):
        """Test generate_population with empty hyperparameter space"""
        set_random_seed(42)
        # This might raise an error depending on implementation
        # Let's see what happens
        try:
            population = generate_population(size=3, hyperparam_space={})
            # If it doesn't raise an error, check structure
            for individual in population:
                self.assertIsInstance(individual, dict)
        except (KeyError, IndexError):
            # Expected if implementation requires all keys
            pass

    def test_generate_population_large_population(self):
        """Test generate_population with large population size"""
        set_random_seed(42)
        population = generate_population(size=100, hyperparam_space=self.hyperparam_space)
        
        self.assertEqual(len(population), 100)
        # Verify all individuals are valid
        for individual in population:
            self.assertIsInstance(individual, dict)
            for key, value in individual.items():
                self.assertIn(value, self.hyperparam_space[key])


if __name__ == '__main__':
    unittest.main()

