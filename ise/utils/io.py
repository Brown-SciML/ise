"""Runtime type-checking utilities for the ise package.

This module provides ``check_type``, a thin wrapper around ``isinstance`` that
raises a descriptive ``TypeError`` when an argument does not match the expected
type.  It is used at the boundaries of public-facing functions to give clear
error messages rather than cryptic internal AttributeErrors::

    from ise.utils.io import check_type

    def process(data, grid):
        check_type(data, pd.DataFrame)
        check_type(grid, (str, xr.Dataset))
        ...
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
