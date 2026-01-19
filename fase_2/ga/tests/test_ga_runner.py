import unittest
from unittest.mock import patch, MagicMock, call
import numpy as np

from fase_2.ga.ga_runner import run_ga
from fase_2.ga.config import HYPERPARAM_SPACE


class TestGARunner(unittest.TestCase):
    """Test cases for GA runner"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "population_size": 5,
            "generations": 2,
            "elite_size": 1,
            "tournament_size": 2,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "random_seed": 42,
            "hyperparam_space": HYPERPARAM_SPACE
        }

    @patch('fase_2.ga.ga_runner.set_random_seed')
    @patch('fase_2.ga.ga_runner.load_dataset')
    @patch('fase_2.ga.ga_runner.prepare_dataset')
    @patch('fase_2.ga.ga_runner.generate_population')
    @patch('fase_2.ga.ga_runner.train_mlp')
    @patch('fase_2.ga.ga_runner.calculate_fitness')
    @patch('fase_2.ga.ga_runner.apply_selection')
    @patch('fase_2.ga.ga_runner.apply_crossover')
    @patch('fase_2.ga.ga_runner.apply_mutation')
    def test_run_ga_basic_flow(
        self,
        mock_mutation,
        mock_crossover,
        mock_selection,
        mock_fitness,
        mock_train_mlp,
        mock_generate_population,
        mock_prepare_dataset,
        mock_load_dataset,
        mock_set_random_seed
    ):
        """Test basic flow of run_ga"""
        # Setup mocks
        mock_load_dataset.return_value = MagicMock()
        mock_prepare_dataset.return_value = MagicMock()
        
        # Create mock population
        mock_population = [
            {"n_hidden_layers": 1, "n_neurons": 16, "learning_rate_init": 0.001, "activation": "relu"},
            {"n_hidden_layers": 2, "n_neurons": 32, "learning_rate_init": 0.01, "activation": "tanh"},
        ]
        mock_generate_population.return_value = mock_population
        
        # Mock model training
        mock_model = MagicMock()
        mock_X_val = np.array([[1, 2], [3, 4]])
        mock_y_val = np.array([0, 1])
        mock_train_mlp.return_value = (mock_model, mock_X_val, mock_y_val)
        
        # Mock fitness calculation
        mock_fitness.return_value = 0.75
        
        # Mock selection
        mock_elites = [mock_population[0]]
        mock_selected = mock_population
        mock_selection.return_value = (mock_elites, mock_selected)
        
        # Mock crossover
        mock_offspring = [mock_population[1]]
        mock_crossover.return_value = mock_offspring
        
        # Mock mutation
        mock_mutation.return_value = mock_offspring
        
        # Run GA
        result = run_ga(self.config)
        
        # Verify calls
        mock_set_random_seed.assert_called_once_with(42)
        mock_load_dataset.assert_called_once()
        mock_prepare_dataset.assert_called_once()
        mock_generate_population.assert_called_once_with(5, HYPERPARAM_SPACE)
        
        # Verify structure of result
        self.assertIn("best_fitness", result)
        self.assertIn("best_individual", result)
        self.assertIn("fitness_history", result)
        self.assertEqual(len(result["fitness_history"]), 2)  # 2 generations

    @patch('fase_2.ga.ga_runner.set_random_seed')
    @patch('fase_2.ga.ga_runner.load_dataset')
    @patch('fase_2.ga.ga_runner.prepare_dataset')
    @patch('fase_2.ga.ga_runner.generate_population')
    @patch('fase_2.ga.ga_runner.train_mlp')
    @patch('fase_2.ga.ga_runner.calculate_fitness')
    @patch('fase_2.ga.ga_runner.apply_selection')
    @patch('fase_2.ga.ga_runner.apply_crossover')
    @patch('fase_2.ga.ga_runner.apply_mutation')
    def test_run_ga_tracks_best_individual(
        self,
        mock_mutation,
        mock_crossover,
        mock_selection,
        mock_fitness,
        mock_train_mlp,
        mock_generate_population,
        mock_prepare_dataset,
        mock_load_dataset,
        mock_set_random_seed
    ):
        """Test that run_ga tracks the best individual across generations"""
        # Setup mocks
        mock_load_dataset.return_value = MagicMock()
        mock_prepare_dataset.return_value = MagicMock()
        
        # Create population matching config size (5 individuals)
        mock_population = [
            {"n_hidden_layers": 1, "n_neurons": 16, "learning_rate_init": 0.001, "activation": "relu"},
            {"n_hidden_layers": 2, "n_neurons": 32, "learning_rate_init": 0.01, "activation": "tanh"},
            {"n_hidden_layers": 1, "n_neurons": 64, "learning_rate_init": 0.1, "activation": "relu"},
            {"n_hidden_layers": 2, "n_neurons": 16, "learning_rate_init": 0.001, "activation": "tanh"},
            {"n_hidden_layers": 1, "n_neurons": 32, "learning_rate_init": 0.01, "activation": "relu"},
        ]
        mock_generate_population.return_value = mock_population
        
        mock_model = MagicMock()
        mock_X_val = np.array([[1, 2]])
        mock_y_val = np.array([0])
        mock_train_mlp.return_value = (mock_model, mock_X_val, mock_y_val)
        
        # With 5 individuals and 2 generations, fitness is called 10 times total
        # Generation 1: all have fitness 0.5, Generation 2: all have fitness 0.9
        mock_fitness.side_effect = [0.5] * 5 + [0.9] * 5
        
        mock_elites = [mock_population[0]]
        # Return the same population structure for selection
        mock_selection.return_value = (mock_elites, mock_population)
        mock_crossover.return_value = mock_population[1:]  # 4 offspring
        mock_mutation.return_value = mock_population[1:]
        
        result = run_ga(self.config)
        
        # Best fitness should be 0.9 (from second generation)
        self.assertEqual(result["best_fitness"], 0.9)
        self.assertIsNotNone(result["best_individual"])

    @patch('fase_2.ga.ga_runner.set_random_seed')
    @patch('fase_2.ga.ga_runner.load_dataset')
    @patch('fase_2.ga.ga_runner.prepare_dataset')
    @patch('fase_2.ga.ga_runner.generate_population')
    @patch('fase_2.ga.ga_runner.train_mlp')
    @patch('fase_2.ga.ga_runner.calculate_fitness')
    @patch('fase_2.ga.ga_runner.apply_selection')
    @patch('fase_2.ga.ga_runner.apply_crossover')
    @patch('fase_2.ga.ga_runner.apply_mutation')
    def test_run_ga_fitness_history(
        self,
        mock_mutation,
        mock_crossover,
        mock_selection,
        mock_fitness,
        mock_train_mlp,
        mock_generate_population,
        mock_prepare_dataset,
        mock_load_dataset,
        mock_set_random_seed
    ):
        """Test that run_ga maintains fitness history"""
        # Setup mocks
        mock_load_dataset.return_value = MagicMock()
        mock_prepare_dataset.return_value = MagicMock()
        
        # Create population matching config size (5 individuals)
        mock_population = [
            {"n_hidden_layers": 1, "n_neurons": 16, "learning_rate_init": 0.001, "activation": "relu"},
            {"n_hidden_layers": 2, "n_neurons": 32, "learning_rate_init": 0.01, "activation": "tanh"},
            {"n_hidden_layers": 1, "n_neurons": 64, "learning_rate_init": 0.1, "activation": "relu"},
            {"n_hidden_layers": 2, "n_neurons": 16, "learning_rate_init": 0.001, "activation": "tanh"},
            {"n_hidden_layers": 1, "n_neurons": 32, "learning_rate_init": 0.01, "activation": "relu"},
        ]
        mock_generate_population.return_value = mock_population
        
        mock_model = MagicMock()
        mock_X_val = np.array([[1, 2]])
        mock_y_val = np.array([0])
        mock_train_mlp.return_value = (mock_model, mock_X_val, mock_y_val)
        
        # Different fitness values for each generation
        # With 5 individuals and 3 generations, fitness is called 15 times total
        # Generation 1: all 0.5, Generation 2: all 0.7, Generation 3: all 0.8
        mock_fitness.side_effect = [0.5] * 5 + [0.7] * 5 + [0.8] * 5
        
        mock_elites = [mock_population[0]]
        mock_selection.return_value = (mock_elites, mock_population)
        mock_crossover.return_value = mock_population[1:]  # 4 offspring
        mock_mutation.return_value = mock_population[1:]
        
        config = self.config.copy()
        config["generations"] = 3
        result = run_ga(config)
        
        # Fitness history should have 3 entries (one per generation, max fitness per generation)
        self.assertEqual(len(result["fitness_history"]), 3)
        self.assertEqual(result["fitness_history"][0], 0.5)
        self.assertEqual(result["fitness_history"][1], 0.7)
        self.assertEqual(result["fitness_history"][2], 0.8)

    @patch('fase_2.ga.ga_runner.set_random_seed')
    @patch('fase_2.ga.ga_runner.load_dataset')
    @patch('fase_2.ga.ga_runner.prepare_dataset')
    @patch('fase_2.ga.ga_runner.generate_population')
    @patch('fase_2.ga.ga_runner.train_mlp')
    @patch('fase_2.ga.ga_runner.calculate_fitness')
    @patch('fase_2.ga.ga_runner.apply_selection')
    @patch('fase_2.ga.ga_runner.apply_crossover')
    @patch('fase_2.ga.ga_runner.apply_mutation')
    def test_run_ga_calls_selection_with_correct_params(
        self,
        mock_mutation,
        mock_crossover,
        mock_selection,
        mock_fitness,
        mock_train_mlp,
        mock_generate_population,
        mock_prepare_dataset,
        mock_load_dataset,
        mock_set_random_seed
    ):
        """Test that run_ga calls selection with correct parameters"""
        # Setup mocks
        mock_load_dataset.return_value = MagicMock()
        mock_prepare_dataset.return_value = MagicMock()
        
        mock_population = [
            {"n_hidden_layers": 1, "n_neurons": 16, "learning_rate_init": 0.001, "activation": "relu"},
        ]
        mock_generate_population.return_value = mock_population
        
        mock_model = MagicMock()
        mock_X_val = np.array([[1, 2]])
        mock_y_val = np.array([0])
        mock_train_mlp.return_value = (mock_model, mock_X_val, mock_y_val)
        mock_fitness.return_value = 0.5
        
        mock_elites = [mock_population[0]]
        mock_selection.return_value = (mock_elites, mock_population)
        mock_crossover.return_value = mock_population
        mock_mutation.return_value = mock_population
        
        run_ga(self.config)
        
        # Verify selection was called with correct parameters
        mock_selection.assert_called()
        # Check that it was called with population_size, elite_size, tournament_size
        calls = mock_selection.call_args_list
        for call_args in calls:
            args, kwargs = call_args
            # First arg should be population_with_fitness (list of tuples)
            self.assertIsInstance(args[0], list)
            self.assertEqual(args[1], self.config["population_size"])
            self.assertEqual(args[2], self.config["elite_size"])
            self.assertEqual(args[3], self.config["tournament_size"])

    @patch('fase_2.ga.ga_runner.set_random_seed')
    @patch('fase_2.ga.ga_runner.load_dataset')
    @patch('fase_2.ga.ga_runner.prepare_dataset')
    @patch('fase_2.ga.ga_runner.generate_population')
    @patch('fase_2.ga.ga_runner.train_mlp')
    @patch('fase_2.ga.ga_runner.calculate_fitness')
    @patch('fase_2.ga.ga_runner.apply_selection')
    @patch('fase_2.ga.ga_runner.apply_crossover')
    @patch('fase_2.ga.ga_runner.apply_mutation')
    def test_run_ga_calls_crossover_with_correct_params(
        self,
        mock_mutation,
        mock_crossover,
        mock_selection,
        mock_fitness,
        mock_train_mlp,
        mock_generate_population,
        mock_prepare_dataset,
        mock_load_dataset,
        mock_set_random_seed
    ):
        """Test that run_ga calls crossover with correct parameters"""
        # Setup mocks
        mock_load_dataset.return_value = MagicMock()
        mock_prepare_dataset.return_value = MagicMock()
        
        mock_population = [
            {"n_hidden_layers": 1, "n_neurons": 16, "learning_rate_init": 0.001, "activation": "relu"},
        ]
        mock_generate_population.return_value = mock_population
        
        mock_model = MagicMock()
        mock_X_val = np.array([[1, 2]])
        mock_y_val = np.array([0])
        mock_train_mlp.return_value = (mock_model, mock_X_val, mock_y_val)
        mock_fitness.return_value = 0.5
        
        mock_elites = [mock_population[0]]
        mock_selection.return_value = (mock_elites, mock_population)
        mock_crossover.return_value = mock_population
        mock_mutation.return_value = mock_population
        
        run_ga(self.config)
        
        # Verify crossover was called
        mock_crossover.assert_called()
        # Check parameters (note: actual signature may vary)
        calls = mock_crossover.call_args_list
        for call_args in calls:
            args, kwargs = call_args
            # Should be called with selected_population
            self.assertIsInstance(args[0], list)

    @patch('fase_2.ga.ga_runner.set_random_seed')
    @patch('fase_2.ga.ga_runner.load_dataset')
    @patch('fase_2.ga.ga_runner.prepare_dataset')
    @patch('fase_2.ga.ga_runner.generate_population')
    @patch('fase_2.ga.ga_runner.train_mlp')
    @patch('fase_2.ga.ga_runner.calculate_fitness')
    @patch('fase_2.ga.ga_runner.apply_selection')
    @patch('fase_2.ga.ga_runner.apply_crossover')
    @patch('fase_2.ga.ga_runner.apply_mutation')
    def test_run_ga_calls_mutation_with_correct_params(
        self,
        mock_mutation,
        mock_crossover,
        mock_selection,
        mock_fitness,
        mock_train_mlp,
        mock_generate_population,
        mock_prepare_dataset,
        mock_load_dataset,
        mock_set_random_seed
    ):
        """Test that run_ga calls mutation with correct parameters"""
        # Setup mocks
        mock_load_dataset.return_value = MagicMock()
        mock_prepare_dataset.return_value = MagicMock()
        
        mock_population = [
            {"n_hidden_layers": 1, "n_neurons": 16, "learning_rate_init": 0.001, "activation": "relu"},
        ]
        mock_generate_population.return_value = mock_population
        
        mock_model = MagicMock()
        mock_X_val = np.array([[1, 2]])
        mock_y_val = np.array([0])
        mock_train_mlp.return_value = (mock_model, mock_X_val, mock_y_val)
        mock_fitness.return_value = 0.5
        
        mock_elites = [mock_population[0]]
        mock_selection.return_value = (mock_elites, mock_population)
        mock_crossover.return_value = mock_population
        mock_mutation.return_value = mock_population
        
        run_ga(self.config)
        
        # Verify mutation was called
        mock_mutation.assert_called()
        calls = mock_mutation.call_args_list
        for call_args in calls:
            args, kwargs = call_args
            # Should be called with offspring
            self.assertIsInstance(args[0], list)
            # Should include mutation_rate and hyperparam_space
            if len(args) > 1:
                self.assertEqual(args[1], self.config["mutation_rate"])

    @patch('fase_2.ga.ga_runner.set_random_seed')
    @patch('fase_2.ga.ga_runner.load_dataset')
    @patch('fase_2.ga.ga_runner.prepare_dataset')
    @patch('fase_2.ga.ga_runner.generate_population')
    @patch('fase_2.ga.ga_runner.train_mlp')
    @patch('fase_2.ga.ga_runner.calculate_fitness')
    @patch('fase_2.ga.ga_runner.apply_selection')
    @patch('fase_2.ga.ga_runner.apply_crossover')
    @patch('fase_2.ga.ga_runner.apply_mutation')
    def test_run_ga_zero_generations(
        self,
        mock_mutation,
        mock_crossover,
        mock_selection,
        mock_fitness,
        mock_train_mlp,
        mock_generate_population,
        mock_prepare_dataset,
        mock_load_dataset,
        mock_set_random_seed
    ):
        """Test run_ga with zero generations"""
        config = self.config.copy()
        config["generations"] = 0
        
        mock_load_dataset.return_value = MagicMock()
        mock_prepare_dataset.return_value = MagicMock()
        mock_generate_population.return_value = []
        
        result = run_ga(config)
        
        # Should return empty fitness history
        self.assertEqual(len(result["fitness_history"]), 0)
        self.assertEqual(result["best_fitness"], float("-inf"))
        self.assertIsNone(result["best_individual"])


if __name__ == '__main__':
    unittest.main()

