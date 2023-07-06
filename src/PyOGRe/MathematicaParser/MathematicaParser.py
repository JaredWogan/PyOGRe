from PyOGRe.MathematicaParser.Nodes import (AddNode, AssocNode, DivNode,
                                            EqNode, ExprNode, FunctionNode,
                                            GreekNode, ListNode, MinusNode,
                                            MulNode, PlusNode, PowNode,
                                            StrNode, SubNode)
from PyOGRe.MathematicaParser.Tokens import TokenType


class Parser:
    """
    Parser Class
    """
    def __init__(self, tokens):
        self.tokens = iter(tokens)
        try:
            self.current = next(self.tokens)
        except StopIteration:
            self.current = None
        try:
            self.next = next(self.tokens)
        except StopIteration:
            self.next = None
        try:
            self.nnext = next(self.tokens)
        except StopIteration:
            self.nnext = None

    def advance(self):
        """
        Advances to the next token
        """
        self.current = self.next
        self.next = self.nnext
        try:
            self.nnext = next(self.tokens)
        except StopIteration:
            self.nnext = None

    def parse(self):
        """
        Parses the tokens and returns the root node
        """
        if self.current is None:
            return None

        result = self.expr()

        if self.current is not None:
            raise Exception(f"Unexpected token: {self.current}")

        return result

    def expr(self, assoc=False):
        """
        Creates an expression
        """
        result = self.term()

        while self.current is not None and (self.current.type in (
            TokenType.PLUS,
            TokenType.MINUS,
            TokenType.WHITESPACE,
            TokenType.LCURL,
            TokenType.LASSOC
        ) or self.current.type == TokenType.ARROW and not assoc):
            if self.current.type == TokenType.PLUS:
                self.advance()
                result = AddNode(result, self.term())
            elif self.current.type == TokenType.MINUS:
                self.advance()
                result = SubNode(result, self.term())
            elif self.current.type == TokenType.WHITESPACE:
                self.advance()
            elif self.current.type == TokenType.LCURL:
                self.advance()
                result = ListNode(self.list())
            elif self.current.type == TokenType.LASSOC:
                self.advance()
                result = self.assoc()
            elif self.current.type == TokenType.ARROW:
                self.advance()
                result = EqNode(result, self.expr())

        return result

    def term(self):
        """
        Creates a term
        """
        while self.current.type == TokenType.WHITESPACE:
            self.advance()
        result = self.factor()

        while self.current is not None and self.current.type in (
            TokenType.MULTIPLY,
            TokenType.DIVIDE,
            TokenType.WHITESPACE
        ):
            if self.current.type == TokenType.MULTIPLY:
                self.advance()
                result = MulNode(result, self.factor())
            elif self.current.type == TokenType.DIVIDE:
                self.advance()
                result = DivNode(result, self.factor())
            elif self.current.type == TokenType.WHITESPACE and self.next is not None and self.next.type in (
                TokenType.EXPR,
                TokenType.LPAREN,
                TokenType.LASSOC,
                TokenType.LBRACE,
                TokenType.COMMAND
            ):
                self.advance()
                result = MulNode(result, self.factor())
            elif self.current.type == TokenType.WHITESPACE:
                self.advance()
            else:
                raise Exception(f"Unexpected token: {self.current}")

        return result

    def list(self):
        """
        Creates a list
        """
        result = []

        while self.current.type != TokenType.RCURL:
            while self.current.type == TokenType.WHITESPACE:
                self.advance()
            result.append(self.expr())
            if self.current.type == TokenType.COMMA:
                self.advance()

        return ListNode(result)

    def assoc(self):
        """
        Creates an association / dictionary
        """
        keys, values = [], []

        while self.current.type != TokenType.RASSOC:
            while self.current.type == TokenType.WHITESPACE:
                self.advance()
            key = self.expr(assoc=True)
            while self.current.type == TokenType.WHITESPACE:
                self.advance()
            if self.current.type == TokenType.ARROW:
                self.advance()
                while self.current.type == TokenType.WHITESPACE:
                    self.advance()
                value = self.expr(assoc=True)
            else:
                value = None
            if key and value:
                keys.append(key)
                values.append(value)
            else:
                raise Exception(f"Unexpected token: {self.current}")
            if self.current.type == TokenType.COMMA:
                self.advance()

        return AssocNode(keys, values)

    def factor(self):
        """
        Creates a factor
        """
        result = None
        while self.current.type == TokenType.WHITESPACE:
            self.advance()
        token = self.current

        if token.type == TokenType.STRING:
            self.advance()
            result = StrNode(token.value)

        if token.type == TokenType.LPAREN:
            self.advance()
            result = self.expr()
            while token.type == TokenType.WHITESPACE:
                self.advance()
            if self.current.type != TokenType.RPAREN:
                raise Exception(f"Expected ')' but found {self.current}")
            self.advance()

        if token.type == TokenType.LCURL:
            self.advance()
            result = self.list()
            while token.type == TokenType.WHITESPACE:
                self.advance()
            if self.current.type != TokenType.RCURL:
                raise Exception("Expected '" + r"}" + f"' but found {token}")
            self.advance()

        if token.type == TokenType.LASSOC:
            self.advance()
            result = self.assoc()
            while token.type == TokenType.WHITESPACE:
                self.advance()
            if self.current.type != TokenType.RASSOC:
                raise Exception(f"Expected '|>' but found {token}")
            self.advance()

        if token.type == TokenType.COMMAND:
            self.advance()
            if self.current.type == TokenType.LBRACE and self.nnext.type == TokenType.RBRACE:
                self.advance()
                result = GreekNode(self.current.value)  # type: ignore[assignment]
                self.advance()
                self.advance()

        if token.type == TokenType.EXPR:
            if self.next is not None and self.next.type == TokenType.LBRACE:
                result = self.func()
            else:
                self.advance()
                result = ExprNode(token.value)  # type: ignore[assignment]

        if token.type == TokenType.PLUS:
            self.advance()
            result = PlusNode(self.factor())  # type: ignore[assignment]

        if token.type == TokenType.MINUS:
            self.advance()
            result = MinusNode(self.factor())  # type: ignore[assignment]

        if self.current is not None and self.current.type == TokenType.POWER:
            self.advance()
            result = PowNode(result, self.factor())  # type: ignore[assignment]
        if self.current is not None and self.next is not None and self.current.type == TokenType.WHITESPACE and self.next.type == TokenType.POWER:
            self.advance()
            self.advance()
            result = PowNode(result, self.factor())  # type: ignore[assignment]

        return result if result else None

    def func(self):
        """
        Creates a function
        """
        func = self.current.value
        args = []
        self.advance()
        while self.current is not None and self.current.type == TokenType.LBRACE:
            self.advance()
            arg = self.expr()
            while self.current.type == TokenType.COMMA:
                self.advance()
                arg = [arg] if not isinstance(arg, list) else arg
                arg.append(self.expr())
            args.append(arg)
            if self.current.type == TokenType.RBRACE:
                self.advance()

        return FunctionNode(func, args)
