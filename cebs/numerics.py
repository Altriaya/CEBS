"""Resolve CEBS numerical primitives from this package, not an asset workspace."""
from pathlib import Path

NUMERICS = Path(__file__).resolve().parent/'_numerics'


def numerical_source(relative):
    path=(NUMERICS/relative).resolve()
    if not path.is_relative_to(NUMERICS.resolve()) or not path.is_file():
        raise FileNotFoundError('Missing CEBS numerical primitive: '+str(relative))
    return path


def bootstrap():
    """Compatibility hook; helpers now use ordinary package imports."""
