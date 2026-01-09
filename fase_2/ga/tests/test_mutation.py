import unittest
from unittest.mock import patch, MagicMock
import random

from fase_2.ga.mutation import mutate, apply_mutation
from fase_2.ga.utils import set_random_seed


class TestMutation(unittest.TestCase):
    """Test cases for mutation operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.hyperparam_space = {
            "n_hidden_layers": [1, 2],
            "n_neurons": [16, 32, 64],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "activation": ["relu", "tanh"]
        }
        self.individual = {
            "n_hidden_layers": 1,
            "n_neurons": 16,
            "learning_rate_init": 0.001,
            "activation": "relu"
        }

    def test_mutate_no_mutation_when_rate_zero(self):
        """Test that no mutation occurs when mutation_rate is 0"""
        set_random_seed(42)
        mutated = mutate(self.individual, mutation_rate=0.0, hyperparam_space=self.hyperparam_space)
        self.assertEqual(mutated, self.individual)

    def test_mutate_always_mutates_when_rate_one(self):
        """Test that mutation always occurs when mutation_rate is 1.0"""
        set_random_seed(42)
        mutated = mutate(self.individual, mutation_rate=1.0, hyperparam_space=self.hyperparam_space)
        # At least one gene should be different
        self.assertNotEqual(mutated, self.individual)

    def test_mutate_preserves_structure(self):
        """Test that mutation preserves the structure of the individual"""
        set_random_seed(42)
        mutated = mutate(self.individual, mutation_rate=0.5, hyperparam_space=self.hyperparam_space)
        self.assertEqual(set(mutated.keys()), set(self.individual.keys()))

    def test_mutate_values_in_valid_space(self):
        """Test that mutated values are within the hyperparameter space"""
        set_random_seed(42)
        mutated = mutate(self.individual, mutation_rate=1.0, hyperparam_space=self.hyperparam_space)
        
        for gene, value in mutated.items():
            self.assertIn(value, self.hyperparam_space[gene])

    def test_mutate_does_not_modify_original(self):
        """Test that mutation does not modify the original individual"""
        set_random_seed(42)
        original = self.individual.copy()
        mutate(self.individual, mutation_rate=1.0, hyperparam_space=self.hyperparam_space)
        self.assertEqual(self.individual, original)

    def test_mutate_with_empty_hyperparam_space(self):
        """Test mutation with empty hyperparameter space"""
        set_random_seed(42)
        mutated = mutate(self.individual, mutation_rate=1.0, hyperparam_space={})
        self.assertEqual(mutated, self.individual)

    @patch('fase_2.ga.mutation.random.random')
    @patch('fase_2.ga.mutation.random.choice')
    def test_mutate_probabilistic_behavior(self, mock_choice, mock_random):
        """Test that mutation follows probabilistic behavior"""
        # Mock random.random to return values that trigger mutation
        mock_random.side_effect = [0.05, 0.15, 0.25, 0.35]  # All below 0.5, so mutations occur
        mock_choice.side_effect = [2, 32, 0.01, "tanh"]
        
        mutated = mutate(self.individual, mutation_rate=0.5, hyperparam_space=self.hyperparam_space)
        
        # Verify random.choice was called for each gene
        self.assertEqual(mock_choice.call_count, 4)

    def test_apply_mutation_empty_population(self):
        """Test apply_mutation with empty population"""
        result = apply_mutation([], mutation_rate=0.1, hyperparam_space=self.hyperparam_space)
        self.assertEqual(result, [])

    def test_apply_mutation_single_individual(self):
        """Test apply_mutation with single individual"""
        set_random_seed(42)
        population = [self.individual]
        result = apply_mutation(population, mutation_rate=0.1, hyperparam_space=self.hyperparam_space)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0].keys()), set(self.individual.keys()))

    def test_apply_mutation_multiple_individuals(self):
        """Test apply_mutation with multiple individuals"""
        set_random_seed(42)
        population = [
            self.individual,
            {
                "n_hidden_layers": 2,
                "n_neurons": 32,
                "learning_rate_init": 0.01,
                "activation": "tanh"
            }
        ]
        result = apply_mutation(population, mutation_rate=0.1, hyperparam_space=self.hyperparam_space)
        
        self.assertEqual(len(result), 2)
        for ind in result:
            self.assertEqual(set(ind.keys()), set(self.individual.keys()))

    def test_apply_mutation_preserves_population_size(self):
        """Test that apply_mutation preserves population size"""
        set_random_seed(42)
        population = [self.individual.copy() for _ in range(5)]
        result = apply_mutation(population, mutation_rate=0.1, hyperparam_space=self.hyperparam_space)
        
        self.assertEqual(len(result), len(population))

    def test_apply_mutation_does_not_modify_original_population(self):
        """Test that apply_mutation does not modify original population"""
        set_random_seed(42)
        population = [self.individual.copy() for _ in range(3)]
        original_population = [ind.copy() for ind in population]
        
        apply_mutation(population, mutation_rate=1.0, hyperparam_space=self.hyperparam_space)
        
        for orig, mod in zip(original_population, population):
            self.assertEqual(orig, mod)


if __name__ == '__main__':
    unittest.main()

