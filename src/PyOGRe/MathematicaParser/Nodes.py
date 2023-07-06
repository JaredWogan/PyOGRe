from dataclasses import dataclass
from typing import Any, List


@dataclass
class AddNode:
    """
    Addition node
    """
    node_a: Any
    node_b: Any

    def __repr__(self):
        return f"({self.node_a} + {self.node_b})"


@dataclass
class SubNode:
    """
    Subtraction node
    """
    node_a: Any
    node_b: Any

    def __repr__(self):
        return f"({self.node_a} - {self.node_b})"


@dataclass
class MulNode:
    """
    Multiplication node
    """
    node_a: Any
    node_b: Any

    def __repr__(self):
        return f"({self.node_a} * {self.node_b})"


@dataclass
class DivNode:
    """
    Division node
    """
    node_a: Any
    node_b: Any

    def __repr__(self):
        return f"({self.node_a} / {self.node_b})"


@dataclass
class PowNode:
    """
    Exponent node
    """
    node_a: Any
    node_b: Any

    def __repr__(self):
        return f"({self.node_a} ^ {self.node_b})"


@dataclass
class PlusNode:
    """
    Plus node
    """
    node: Any

    def __repr__(self):
        return "".join(["+(", str(self.node), ")"])


@dataclass
class MinusNode:
    """
    Minus node
    """
    node: Any

    def __repr__(self):
        return "".join(["-(", str(self.node), ")"])


@dataclass
class StrNode:
    """
    String node
    """
    value: str

    def __repr__(self):
        return "".join(["(", self.value, ")"])


@dataclass
class ExprNode:
    """
    Expression node
    """
    expr: Any

    def __repr__(self):
        return f"{self.expr}"


@dataclass
class EqNode:
    """
    Equation Node
    """
    node_a: Any
    node_b: Any

    def __repr__(self):
        return f"{self.node_a} = {self.node_b}"


@dataclass
class ListNode:
    """
    List node
    """
    array: List[Any]

    def __repr__(self):
        return f"{self.array}"


@dataclass
class AssocNode:
    """
    Association node
    """
    keys: List[Any]
    values: List[Any]

    def __repr__(self):
        return f"({self.keys}: {self.values})"


@dataclass
class GreekNode:
    """
    Greek node
    """
    value: str

    def __repr__(self):
        return f"GREEK:{self.value}"


@dataclass
class FunctionNode:
    """
    Division node
    """
    func: Any
    args: Any

    def __repr__(self):
        return f"({self.func}{self.args})"
