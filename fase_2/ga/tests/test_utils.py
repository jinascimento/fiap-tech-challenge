import unittest
import random
import numpy as np

from fase_2.ga.utils import set_random_seed


class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""

    def test_set_random_seed_reproducibility_random(self):
        """Test that set_random_seed makes random module reproducible"""
        set_random_seed(42)
        value1 = random.random()
        
        set_random_seed(42)
        value2 = random.random()
        
        self.assertEqual(value1, value2)

    def test_set_random_seed_reproducibility_numpy(self):
        """Test that set_random_seed makes numpy random reproducible"""
        set_random_seed(42)
        value1 = np.random.random()
        
        set_random_seed(42)
        value2 = np.random.random()
        
        self.assertEqual(value1, value2)

    def test_set_random_seed_different_seeds_different_values(self):
        """Test that different seeds produce different values"""
        set_random_seed(42)
        value1 = random.random()
        
        set_random_seed(43)
        value2 = random.random()
        
        # They should be different (very high probability)
        self.assertNotEqual(value1, value2)

    def test_set_random_seed_affects_both_modules(self):
        """Test that set_random_seed affects both random and numpy.random"""
        set_random_seed(42)
        random_value = random.random()
        numpy_value = np.random.random()
        
        set_random_seed(42)
        random_value2 = random.random()
        numpy_value2 = np.random.random()
        
        self.assertEqual(random_value, random_value2)
        self.assertEqual(numpy_value, numpy_value2)

    def test_set_random_seed_with_zero(self):
        """Test set_random_seed with seed=0"""
        set_random_seed(0)
        value1 = random.random()
        
        set_random_seed(0)
        value2 = random.random()
        
        self.assertEqual(value1, value2)

    def test_set_random_seed_with_negative_seed(self):
        """Test set_random_seed with negative seed"""
        # numpy doesn't accept negative seeds, should raise ValueError
        with self.assertRaises(ValueError):
            set_random_seed(-1)

    def test_set_random_seed_with_large_seed(self):
        """Test set_random_seed with large seed value"""
        set_random_seed(999999)
        value1 = random.random()
        
        set_random_seed(999999)
        value2 = random.random()
        
        self.assertEqual(value1, value2)

    def test_set_random_seed_sequence_reproducibility(self):
        """Test that set_random_seed makes sequences reproducible"""
        set_random_seed(42)
        sequence1 = [random.random() for _ in range(10)]
        
        set_random_seed(42)
        sequence2 = [random.random() for _ in range(10)]
        
        self.assertEqual(sequence1, sequence2)

    def test_set_random_seed_numpy_sequence_reproducibility(self):
        """Test that set_random_seed makes numpy sequences reproducible"""
        set_random_seed(42)
        sequence1 = [np.random.random() for _ in range(10)]
        
        set_random_seed(42)
        sequence2 = [np.random.random() for _ in range(10)]
        
        np.testing.assert_array_equal(sequence1, sequence2)


if __name__ == '__main__':
    unittest.main()

