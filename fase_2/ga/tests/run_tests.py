#!/usr/bin/env python
"""
Test runner for GA algorithm tests.
Run all tests with: python -m fase_2.ga.tests.run_tests
Or run individual test files with: python -m unittest fase_2.ga.tests.test_mutation
"""

import unittest
import sys
import os

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

if __name__ == '__main__':
    # Discover and run all tests in the tests directory
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(__file__), pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

