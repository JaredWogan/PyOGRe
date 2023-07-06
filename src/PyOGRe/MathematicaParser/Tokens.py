from dataclasses import dataclass
from enum import Enum
from typing import Any


class TokenType(Enum):
    """
    Token Types Class
    """
    EXPR = 0
    PLUS = 1
    MINUS = 2
    MULTIPLY = 3
    DIVIDE = 4
    POWER = 5
    LPAREN = 6
    RPAREN = 7
    LBRACE = 8
    RBRACE = 9
    LCURL = 10
    RCURL = 11
    LASSOC = 12
    RASSOC = 13
    ARROW = 14
    COMMAND = 15
    WHITESPACE = 16
    STRING = 17
    COMMA = 18


@dataclass
class Token:
    """
    Token Class
    """
    type: TokenType
    value: Any = None

    def __repr__(self):
        return self.type.name + (f":{self.value}" if self.value is not None else "")
