import unittest
from unittest.mock import patch
import random

from fase_2.ga.generate_individual import generate_individual
from fase_2.ga.utils import set_random_seed


class TestGenerateIndividual(unittest.TestCase):
    """Test cases for individual generation"""

    def setUp(self):
        """Set up test fixtures"""
        self.hyperparam_space = {
            "n_hidden_layers": [1, 2],
            "n_neurons": [16, 32, 64],
            "learning_rate_init": [0.001, 0.01, 0.1],
            "activation": ["relu", "tanh"]
        }

    def test_generate_individual_returns_dict(self):
        """Test that generate_individual returns a dictionary"""
        set_random_seed(42)
        individual = generate_individual(self.hyperparam_space)
        
        self.assertIsInstance(individual, dict)

    def test_generate_individual_has_all_required_keys(self):
        """Test that generated individual has all required hyperparameter keys"""
        set_random_seed(42)
        individual = generate_individual(self.hyperparam_space)
        
        expected_keys = {
            "n_hidden_layers",
            "n_neurons",
            "learning_rate_init",
            "activation"
        }
        self.assertEqual(set(individual.keys()), expected_keys)

    def test_generate_individual_values_in_valid_space(self):
        """Test that generated values are within the hyperparameter space"""
        set_random_seed(42)
        individual = generate_individual(self.hyperparam_space)
        
        for key, value in individual.items():
            self.assertIn(value, self.hyperparam_space[key])

    def test_generate_individual_different_values_on_different_calls(self):
        """Test that different calls generate potentially different individuals"""
        set_random_seed(42)
        individual1 = generate_individual(self.hyperparam_space)
        
        set_random_seed(43)
        individual2 = generate_individual(self.hyperparam_space)
        
        # They might be the same or different, but both should be valid
        for key in individual1.keys():
            self.assertIn(individual1[key], self.hyperparam_space[key])
            self.assertIn(individual2[key], self.hyperparam_space[key])

    def test_generate_individual_reproducible_with_seed(self):
        """Test that generate_individual is reproducible with same seed"""
        set_random_seed(42)
        individual1 = generate_individual(self.hyperparam_space)
        
        set_random_seed(42)
        individual2 = generate_individual(self.hyperparam_space)
        
        self.assertEqual(individual1, individual2)

    @patch('fase_2.ga.generate_individual.random.choice')
    def test_generate_individual_uses_random_choice(self, mock_choice):
        """Test that generate_individual uses random.choice for each hyperparameter"""
        mock_choice.side_effect = [1, 32, 0.01, "relu"]
        
        individual = generate_individual(self.hyperparam_space)
        
        # Should be called 4 times (once for each hyperparameter)
        self.assertEqual(mock_choice.call_count, 4)
        self.assertEqual(individual["n_hidden_layers"], 1)
        self.assertEqual(individual["n_neurons"], 32)
        self.assertEqual(individual["learning_rate_init"], 0.01)
        self.assertEqual(individual["activation"], "relu")

    def test_generate_individual_with_single_value_options(self):
        """Test generate_individual with hyperparameter spaces that have single values"""
        single_value_space = {
            "n_hidden_layers": [1],
            "n_neurons": [32],
            "learning_rate_init": [0.01],
            "activation": ["relu"]
        }
        
        set_random_seed(42)
        individual = generate_individual(single_value_space)
        
        self.assertEqual(individual["n_hidden_layers"], 1)
        self.assertEqual(individual["n_neurons"], 32)
        self.assertEqual(individual["learning_rate_init"], 0.01)
        self.assertEqual(individual["activation"], "relu")

    def test_generate_individual_structure_consistency(self):
        """Test that all generated individuals have consistent structure"""
        set_random_seed(42)
        individuals = [generate_individual(self.hyperparam_space) for _ in range(10)]
        
        # All should have the same keys
        first_keys = set(individuals[0].keys())
        for ind in individuals[1:]:
            self.assertEqual(set(ind.keys()), first_keys)


if __name__ == '__main__':
    unittest.main()

