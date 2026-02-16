import os
import sys

# Garantir que a raiz do projeto esteja no PYTHONPATH para imports do pacote `src`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
