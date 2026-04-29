"""Type-checking and I/O utilities for the ise package.

This module provides check_type for validating argument types at runtime.
"""


def check_type(obj, types):
    """
    Validate that an object is an instance of the given type(s).

    Args:
        obj: Object to check.
        types: Expected type or tuple of types.

    Returns:
        int: 1 if validation passes (for use in conditional logic).

    Raises:
        TypeError: If obj is not an instance of types.
    """
    if not isinstance(obj, types):
        raise TypeError(f"Expected {types}, got {type(obj).__name__}")
    return 1
