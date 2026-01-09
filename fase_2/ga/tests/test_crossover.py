import unittest
from unittest.mock import patch, MagicMock
import random

from fase_2.ga.crossover import crossover, apply_crossover
from fase_2.ga.utils import set_random_seed


class TestCrossover(unittest.TestCase):
    """Test cases for crossover operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.parent1 = {
            "n_hidden_layers": 1,
            "n_neurons": 16,
            "learning_rate_init": 0.001,
            "activation": "relu"
        }
        self.parent2 = {
            "n_hidden_layers": 2,
            "n_neurons": 32,
            "learning_rate_init": 0.01,
            "activation": "tanh"
        }

    def test_crossover_creates_valid_child(self):
        """Test that crossover creates a valid child"""
        set_random_seed(42)
        child = crossover(self.parent1, self.parent2)
        
        self.assertIsInstance(child, dict)
        self.assertEqual(set(child.keys()), set(self.parent1.keys()))

    def test_crossover_child_has_values_from_parents(self):
        """Test that child values come from either parent"""
        set_random_seed(42)
        child = crossover(self.parent1, self.parent2)
        
        for gene in child.keys():
            self.assertIn(child[gene], [self.parent1[gene], self.parent2[gene]])

    def test_crossover_does_not_modify_parents(self):
        """Test that crossover does not modify parent individuals"""
        set_random_seed(42)
        parent1_copy = self.parent1.copy()
        parent2_copy = self.parent2.copy()
        
        crossover(self.parent1, self.parent2)
        
        self.assertEqual(self.parent1, parent1_copy)
        self.assertEqual(self.parent2, parent2_copy)

    def test_crossover_different_parents(self):
        """Test crossover with completely different parents"""
        parent1 = {"gene1": "A", "gene2": 1}
        parent2 = {"gene1": "B", "gene2": 2}
        
        set_random_seed(42)
        child = crossover(parent1, parent2)
        
        self.assertIn(child["gene1"], ["A", "B"])
        self.assertIn(child["gene2"], [1, 2])

    @patch('fase_2.ga.crossover.random.choice')
    def test_crossover_uses_random_choice(self, mock_choice):
        """Test that crossover uses random.choice for each gene"""
        mock_choice.side_effect = [self.parent1["n_hidden_layers"],
                                   self.parent2["n_neurons"],
                                   self.parent1["learning_rate_init"],
                                   self.parent2["activation"]]
        
        child = crossover(self.parent1, self.parent2)
        
        # Should be called 4 times (once for each gene)
        self.assertEqual(mock_choice.call_count, 4)
        self.assertEqual(child["n_hidden_layers"], self.parent1["n_hidden_layers"])
        self.assertEqual(child["n_neurons"], self.parent2["n_neurons"])

    def test_apply_crossover_creates_correct_number_of_offspring(self):
        """Test that apply_crossover creates correct number of offspring"""
        set_random_seed(42)
        parents = [self.parent1, self.parent2]
        
        offspring = apply_crossover(
            parents,
            pop_size=5,
            elite_size=1,
            crossover_rate=0.8
        )
        
        # Should create pop_size - elite_size = 5 - 1 = 4 offspring
        self.assertEqual(len(offspring), 4)

    def test_apply_crossover_offspring_are_valid_individuals(self):
        """Test that offspring are valid individuals"""
        set_random_seed(42)
        parents = [self.parent1, self.parent2]
        
        offspring = apply_crossover(
            parents,
            pop_size=3,
            elite_size=1,
            crossover_rate=0.8
        )
        
        for child in offspring:
            self.assertIsInstance(child, dict)
            self.assertEqual(set(child.keys()), set(self.parent1.keys()))

    def test_apply_crossover_no_elites(self):
        """Test apply_crossover when elite_size=0"""
        set_random_seed(42)
        parents = [self.parent1, self.parent2]
        
        offspring = apply_crossover(
            parents,
            pop_size=4,
            elite_size=0,
            crossover_rate=0.8
        )
        
        # Should create pop_size - 0 = 4 offspring
        self.assertEqual(len(offspring), 4)

    def test_apply_crossover_all_elites(self):
        """Test apply_crossover when elite_size equals pop_size"""
        set_random_seed(42)
        parents = [self.parent1, self.parent2]
        
        offspring = apply_crossover(
            parents,
            pop_size=2,
            elite_size=2,
            crossover_rate=0.8
        )
        
        # Should create pop_size - elite_size = 0 offspring
        self.assertEqual(len(offspring), 0)

    def test_apply_crossover_crossover_rate_zero(self):
        """Test apply_crossover when crossover_rate=0 (no crossover)"""
        set_random_seed(42)
        parents = [self.parent1, self.parent2]
        
        offspring = apply_crossover(
            parents,
            pop_size=3,
            elite_size=1,
            crossover_rate=0.0
        )
        
        # All offspring should be copies of parent1 (since crossover doesn't happen)
        for child in offspring:
            self.assertIsInstance(child, dict)
            # Structure should match
            self.assertEqual(set(child.keys()), set(self.parent1.keys()))

    def test_apply_crossover_crossover_rate_one(self):
        """Test apply_crossover when crossover_rate=1.0 (always crossover)"""
        set_random_seed(42)
        parents = [self.parent1, self.parent2]
        
        offspring = apply_crossover(
            parents,
            pop_size=3,
            elite_size=1,
            crossover_rate=1.0
        )
        
        # All offspring should be crossovers
        for child in offspring:
            self.assertIsInstance(child, dict)
            # Each gene should come from one of the parents
            for gene in child.keys():
                self.assertIn(child[gene], [self.parent1[gene], self.parent2[gene]])

    @patch('fase_2.ga.crossover.random.random')
    @patch('fase_2.ga.crossover.random.choice')
    def test_apply_crossover_probabilistic_behavior(self, mock_choice, mock_random):
        """Test that apply_crossover follows probabilistic behavior"""
        # Mock random.random to return values that trigger crossover
        # Need 3 calls (for 3 offspring)
        mock_random.side_effect = [0.5, 0.7, 0.9]  # All below 0.8, so crossovers occur
        # For each offspring: 2 parent selections + 4 gene selections
        mock_choice.side_effect = [
            self.parent1, self.parent2,  # parent selection for offspring 1
            self.parent1["n_hidden_layers"], self.parent2["n_neurons"], 
            self.parent1["learning_rate_init"], self.parent2["activation"],  # crossover choices
            self.parent1, self.parent2,  # parent selection for offspring 2
            self.parent1["n_hidden_layers"], self.parent2["n_neurons"],
            self.parent1["learning_rate_init"], self.parent2["activation"],  # crossover choices
            self.parent1, self.parent2,  # parent selection for offspring 3
            self.parent1["n_hidden_layers"], self.parent2["n_neurons"],
            self.parent1["learning_rate_init"], self.parent2["activation"],  # crossover choices
        ]
        
        parents = [self.parent1, self.parent2]
        offspring = apply_crossover(
            parents,
            pop_size=4,
            elite_size=1,
            crossover_rate=0.8
        )
        
        self.assertEqual(len(offspring), 3)

    def test_apply_crossover_single_parent(self):
        """Test apply_crossover with single parent"""
        set_random_seed(42)
        parents = [self.parent1]
        
        offspring = apply_crossover(
            parents,
            pop_size=3,
            elite_size=1,
            crossover_rate=0.8
        )
        
        # Should still work, parent1 will be selected for both parent roles
        self.assertEqual(len(offspring), 2)
        for child in offspring:
            self.assertIsInstance(child, dict)

    def test_apply_crossover_empty_parents(self):
        """Test apply_crossover with empty parents list"""
        set_random_seed(42)
        # This should raise an IndexError when trying to select from empty list
        with self.assertRaises(IndexError):
            apply_crossover(
                [],
                pop_size=3,
                elite_size=1,
                crossover_rate=0.8
            )


if __name__ == '__main__':
    unittest.main()

