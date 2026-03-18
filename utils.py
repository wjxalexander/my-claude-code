"""Utility functions for the my-claude-code package."""


def add(a: int | float, b: int | float) -> int | float:
    """Add two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


def subtract(a: int | float, b: int | float) -> int | float:
    """Subtract two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The difference of a and b
    """
    return a - b


def multiply(a: int | float, b: int | float) -> int | float:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The product of a and b
    """
    return a * b


def divide(a: int | float, b: int | float) -> int | float:
    """Divide two numbers.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        The quotient of a and b

    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
