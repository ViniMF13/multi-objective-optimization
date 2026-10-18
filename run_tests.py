# -*- coding: utf-8 -*-
"""
run_tests.py
Script para execução de toda a suíte de testes unitários e de integração do projeto.
Pode ser executado com:
    python run_tests.py
ou
    python -m unittest discover tests
"""

import sys
import unittest


def main():
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
