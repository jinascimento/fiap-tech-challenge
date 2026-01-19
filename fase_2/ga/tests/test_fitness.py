import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from sklearn.metrics import f1_score

from fase_2.ga.fitness import calculate_fitness


class TestFitness(unittest.TestCase):
    """Test cases for fitness calculation"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a mock model
        self.mock_model = MagicMock()
        self.X_test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.y_test = np.array([0, 1, 0])

    def test_calculate_fitness_calls_model_predict(self):
        """Test that calculate_fitness calls model.predict"""
        self.mock_model.predict.return_value = np.array([0, 1, 0])
        
        calculate_fitness(self.mock_model, self.X_test, self.y_test)
        
        self.mock_model.predict.assert_called_once_with(self.X_test)

    def test_calculate_fitness_returns_f1_score(self):
        """Test that calculate_fitness returns f1_score"""
        y_pred = np.array([0, 1, 0])
        self.mock_model.predict.return_value = y_pred
        
        expected_f1 = f1_score(self.y_test, y_pred)
        result = calculate_fitness(self.mock_model, self.X_test, self.y_test)
        
        self.assertEqual(result, expected_f1)

    def test_calculate_fitness_perfect_prediction(self):
        """Test fitness calculation with perfect predictions"""
        y_pred = np.array([0, 1, 0])  # Perfect match
        self.mock_model.predict.return_value = y_pred
        
        result = calculate_fitness(self.mock_model, self.X_test, self.y_test)
        
        self.assertEqual(result, 1.0)

    def test_calculate_fitness_no_correct_predictions(self):
        """Test fitness calculation with no correct predictions"""
        y_pred = np.array([1, 0, 1])  # All wrong
        self.mock_model.predict.return_value = y_pred
        
        result = calculate_fitness(self.mock_model, self.X_test, self.y_test)
        
        # F1 score should be 0 when there are no true positives
        self.assertEqual(result, 0.0)

    def test_calculate_fitness_partial_correct_predictions(self):
        """Test fitness calculation with partial correct predictions"""
        y_pred = np.array([0, 1, 1])  # 2 out of 3 correct
        self.mock_model.predict.return_value = y_pred
        
        expected_f1 = f1_score(self.y_test, y_pred)
        result = calculate_fitness(self.mock_model, self.X_test, self.y_test)
        
        self.assertEqual(result, expected_f1)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_calculate_fitness_different_array_sizes(self):
        """Test fitness calculation with different array sizes"""
        y_pred = np.array([0, 1, 0, 1, 0])
        y_test = np.array([0, 1, 0])
        self.mock_model.predict.return_value = y_pred
        
        # This should raise a ValueError from sklearn
        with self.assertRaises(ValueError):
            calculate_fitness(self.mock_model, self.X_test, y_test)

    def test_calculate_fitness_binary_classification(self):
        """Test fitness calculation for binary classification"""
        y_test = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        self.mock_model.predict.return_value = y_pred
        
        expected_f1 = f1_score(y_test, y_pred)
        result = calculate_fitness(self.mock_model, self.X_test[:len(y_test)], y_test)
        
        self.assertEqual(result, expected_f1)

    def test_calculate_fitness_returns_float(self):
        """Test that calculate_fitness returns a float"""
        y_pred = np.array([0, 1, 0])
        self.mock_model.predict.return_value = y_pred
        
        result = calculate_fitness(self.mock_model, self.X_test, self.y_test)
        
        self.assertIsInstance(result, (float, np.floating))

    def test_calculate_fitness_with_empty_arrays(self):
        """Test fitness calculation with empty arrays"""
        y_test = np.array([])
        y_pred = np.array([])
        X_test = np.array([]).reshape(0, 3)
        self.mock_model.predict.return_value = y_pred
        
        # sklearn returns 0.0 with a warning for empty arrays
        result = calculate_fitness(self.mock_model, X_test, y_test)
        self.assertEqual(result, 0.0)


if __name__ == '__main__':
    unittest.main()

