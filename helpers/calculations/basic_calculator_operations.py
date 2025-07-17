from langchain_core.tools import tool

@tool
def add(a: int | float, b: int | float) -> int | float | None:
    """
    Adds two numbers
    :param a: first number to add
    :param b: second number to add
    :return: the sum of the two numbers
    """
    return a + b


@tool
def subtract(a: int | float, b: int | float) -> int | float | None:
    """
    Subtract one number from another
    :param a: number to subtract from
    :param b: number to be subtracted
    :return: the result of the subtraction of b from a
    """
    return a - b


@tool
def divide(a: int | float, b: int | float) -> int | float | str:
    """
    Divides one number by another
    :param a: number to divide
    :param b: number to divide by
    :return: the result of the division of a by b
    """
    if int(b) == 0:
        return "You cannot divide by 0!"

    return a / b


@tool
def multiply(a: int | float, b: int | float) -> int | float | None:
    """
    Multiply two numbers together.
    :param a: first number to multiply
    :param b: second number to multiply
    :return: the result of the multiplication between the two numbers
    """
    return a * b
