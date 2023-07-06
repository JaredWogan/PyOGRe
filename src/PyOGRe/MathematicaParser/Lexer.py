from PyOGRe.MathematicaParser.Tokens import Token, TokenType


WHITESPACE = " \n\t"
DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
BRACKETS = r"()[]{}"
ASSOC = ["<|", "|>", "->"]
STRING = ['"', "'", '"""']


class Lexer:
    """
    Lexer Class
    """
    def __init__(self, string):
        self.string = iter(string)
        try:
            self.current = next(self.string)
        except StopIteration:
            self.current = None
        try:
            self.next = next(self.string)
        except StopIteration:
            self.next = None
        try:
            self.nnext = next(self.string)
        except StopIteration:
            self.nnext = None

    def advance(self):
        """
        Advances to the next character in the string
        """
        self.current = self.next
        self.next = self.nnext
        try:
            self.nnext = next(self.string)
        except StopIteration:
            self.nnext = None

    def generate_tokens(self):
        """
        Returns a generator of tokens
        """
        while self.current is not None:
            if self.current in WHITESPACE:
                yield Token(TokenType.WHITESPACE, None)
            elif self.current == "." or self.current in DIGITS or self.current in LETTERS:
                yield self.generate_expr()
            elif self.current in STRING or str(self.current) + str(self.next) + str(self.nnext) in STRING:
                yield self.generate_str()
            elif self.current == "+":
                yield Token(TokenType.PLUS, None)
            elif self.current == "-" and self.next != ">":
                yield Token(TokenType.MINUS, None)
            elif self.current == "*":
                yield Token(TokenType.MULTIPLY, None)
            elif self.current == "/":
                yield Token(TokenType.DIVIDE, None)
            elif self.current == "^":
                yield Token(TokenType.POWER, None)
            elif self.current == "(":
                yield Token(TokenType.LPAREN, None)
            elif self.current == ")":
                yield Token(TokenType.RPAREN, None)
            elif self.current == "[":
                yield Token(TokenType.LBRACE, None)
            elif self.current == "{":
                yield Token(TokenType.LCURL, None)
            elif self.current == "}":
                yield Token(TokenType.RCURL, None)
            elif self.current == "]":
                yield Token(TokenType.RBRACE, None)
            elif self.current == "<" and self.next == "|":
                self.advance()
                yield Token(TokenType.LASSOC, None)
            elif self.current == "|" and self.next == ">":
                self.advance()
                yield Token(TokenType.RASSOC, None)
            elif self.current == "-" and self.next == ">":
                self.advance()
                yield Token(TokenType.ARROW, None)
            elif self.current == "\\":
                yield Token(TokenType.COMMAND, None)
            elif self.current in ["<", ">", "|"]:
                self.advance()
            elif self.current == ",":
                yield Token(TokenType.COMMA, None)
            else:
                raise Exception("Unknown character: " + str(self.current) + " " + str(self.next))
            self.advance()

    def generate_expr(self):
        """
        Generate an expression from the current character
        """
        decimal_count = 0
        expr_str = ""

        while (
            self.current is not None and
            self.next is not None and
            (self.next in DIGITS or self.next in LETTERS) and
            self.next not in WHITESPACE and
            self.next not in BRACKETS and
            str(self.next) + str(self.nnext) not in ASSOC and
            self.next != ","
        ):
            if self.current == ".":
                decimal_count += 1
                if decimal_count > 1:
                    break
            expr_str += self.current
            self.advance()

        expr_str += self.current if self.current is not None else ""

        if expr_str.startswith("."):
            expr_str = "0" + expr_str
        if expr_str.endswith("."):
            expr_str += "0"

        return Token(TokenType.EXPR, expr_str)

    def generate_str(self):
        """
        Generate a string from the current character
        """
        string = self.current if str(self.current) + str(self.next) + str(self.nnext) != '"""' else '"""'
        string_str = string
        if string == '"""':
            self.advance()
            self.advance()
        self.advance()

        while (
            self.current is not None and
            self.next is not None and
            self.next != string and
            str(self.current) + str(self.next) + str(self.nnext) != '"""'
        ):
            string_str += self.current
            self.advance()

        string_str += self.current if self.current not in STRING else ""
        string_str += string
        if string == '"""':
            self.advance()
            self.advance()
        self.advance()

        return Token(TokenType.STRING, string_str)
