import sympy as sym


greek_letters = {
    "CapitalAlpha": "Alpha",
    "CapitalBeta": "Beta",
    "CapitalGamma": "Gamma",
    "CapitalDelta": "Delta",
    "CapitalEpsilon": "Epsilon",
    "CapitalZeta": "Zeta",
    "CapitalEta": "Eta",
    "CapitalTheta": "Theta",
    "CapitalIota": "Iota",
    "CapitalKappa": "Kappa",
    "CapitalLambda": "Lambda",
    "CapitalMu": "Mu",
    "CapitalNu": "Nu",
    "CapitalXi": "Xi",
    "CapitalOmicron": "Omicron",
    "CapitalPi": "Pi",
    "CapitalRho": "Rho",
    "CapitalSigma": "Sigma",
    "CapitalTau": "Tau",
    "CapitalUpsilon": "Upsilon",
    "CapitalPhi": "Phi",
    "CapitalChi": "Chi",
    "CapitalPsi": "Psi",
    "CapitalOmega": "Omega",
    "Alpha": "alpha",
    "Beta": "beta",
    "Gamma": "gamma",
    "Delta": "delta",
    "Epsilon": "epsilon",
    "Zeta": "zeta",
    "Eta": "eta",
    "Theta": "theta",
    "Iota": "iota",
    "Kappa": "kappa",
    "Lambda": "lambda",
    "Mu": "mu",
    "Nu": "nu",
    "Xi": "xi",
    "Omicron": "omicron",
    "Pi": "pi",
    "Rho": "rho",
    "Sigma": "sigma",
    "Tau": "tau",
    "Upsilon": "upsilon",
    "Phi": "phi",
    "Chi": "chi",
    "Psi": "psi",
    "Omega": "omega",
    "DottedSquare": "T"
}

greek_commands = {
    r"\[CapitalAlpha]": "Alpha",
    r"\[CapitalBeta]": "Beta",
    r"\[CapitalGamma]": "Gamma",
    r"\[CapitalDelta]": "Delta",
    r"\[CapitalEpsilon]": "Epsilon",
    r"\[CapitalZeta]": "Zeta",
    r"\[CapitalEta]": "Eta",
    r"\[CapitalTheta]": "Theta",
    r"\[CapitalIota]": "Iota",
    r"\[CapitalKappa]": "Kappa",
    r"\[CapitalLambda]": "Lambda",
    r"\[CapitalMu]": "Mu",
    r"\[CapitalNu]": "Nu",
    r"\[CapitalXi]": "Xi",
    r"\[CapitalOmicron]": "Omicron",
    r"\[CapitalPi]": "Pi",
    r"\[CapitalRho]": "Rho",
    r"\[CapitalSigma]": "Sigma",
    r"\[CapitalTau]": "Tau",
    r"\[CapitalUpsilon]": "Upsilon",
    r"\[CapitalPhi]": "Phi",
    r"\[CapitalChi]": "Chi",
    r"\[CapitalPsi]": "Psi",
    r"\[CapitalOmega]": "Omega",
    r"\[Alpha]": "alpha",
    r"\[Beta]": "beta",
    r"\[Gamma]": "gamma",
    r"\[Delta]": "delta",
    r"\[Epsilon]": "epsilon",
    r"\[Zeta]": "zeta",
    r"\[Eta]": "eta",
    r"\[Theta]": "theta",
    r"\[Iota]": "iota",
    r"\[Kappa]": "kappa",
    r"\[Lambda]": "lambda",
    r"\[Mu]": "mu",
    r"\[Nu]": "nu",
    r"\[Xi]": "xi",
    r"\[Omicron]": "omicron",
    r"\[Pi]": "pi",
    r"\[Rho]": "rho",
    r"\[Sigma]": "sigma",
    r"\[Tau]": "tau",
    r"\[Upsilon]": "upsilon",
    r"\[Phi]": "phi",
    r"\[Chi]": "chi",
    r"\[Psi]": "psi",
    r"\[Omega]": "omega",
    r"\[DottedSquare]": ""
}

known_functions = {
    "Sin": sym.sin,
    "Cos": sym.cos,
    "Tan": sym.tan,
    "ArcSin": sym.asin,
    "ArcCos": sym.acos,
    "ArcTan": sym.atan,
    "Sec": sym.sec,
    "Csc": sym.csc,
    "Cot": sym.cot,
    "Sinh": sym.sinh,
    "Cosh": sym.cosh,
    "Tanh": sym.tanh,
    "ArcSinh": sym.asinh,
    "ArcCosh": sym.acosh,
    "ArcTanh": sym.atanh,
    "Sech": sym.sech,
    "Csch": sym.csch,
    "Coth": sym.coth,
    "Exp": sym.exp,
    "Log": sym.log
}


def replace_greek_commands(string):
    """
    Replaces greek letter commands with the greek letter
    """
    for greek_command, greek_letter in greek_commands.items():
        string = string.replace(greek_command, greek_letter)
    return string


def parse_str(string):
    """
    Parses a string containing Mathematica code, returning a SymPy expression
    """
    from PyOGRe.MathematicaParser.Lexer import Lexer
    from PyOGRe.MathematicaParser.MathematicaParser import Parser
    string = string.replace(r"\\ReleaseHold", "")
    string = " ".join(string.split())
    return Interpreter().visit(Parser(Lexer(string).generate_tokens()).parse())


class Interpreter:
    """
    Interpreter Class
    """
    def visit(self, node):
        """
        Calls the visit method of the node
        """
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name)
        return method(node)

    def visit_AddNode(self, node):
        """
        Converts addition node to sympy addition
        """
        return sym.Add(self.visit(node.node_a), self.visit(node.node_b))

    def visit_SubNode(self, node):
        """
        Converts subtraction node to sympy subtraction
        """
        return sym.Add(self.visit(node.node_a), -self.visit(node.node_b))

    def visit_MulNode(self, node):
        """
        Converts multiplication node to sympy multiplication
        """
        return sym.Mul(self.visit(node.node_a), self.visit(node.node_b))

    def visit_DivNode(self, node):
        """
        Converts division node to sympy division
        """
        return sym.Mul(self.visit(node.node_a), sym.Pow(self.visit(node.node_b), -1))

    def visit_PowNode(self, node):
        """
        Converts power node to sympy exponentiation
        """
        return sym.Pow(self.visit(node.node_a), self.visit(node.node_b))

    def visit_PlusNode(self, node):
        """
        Converts plus node to sympy expression
        """
        return sym.sympify(self.visit(node.node))

    def visit_MinusNode(self, node):
        """
        Converts minus node to sympy expression
        """
        return -sym.sympify(self.visit(node.node))

    def visit_StrNode(self, node):
        """
        Converts string node to a string
        """
        from ast import literal_eval
        return literal_eval(replace_greek_commands(node.value))

    def visit_ExprNode(self, node):
        """
        Converts generic expression node to sympy expression
        """
        return sym.sympify(node.expr)

    def visit_ListNode(self, node):
        """
        Converts list node to python list
        """
        return [
            self.visit(child)
            for child in node.array
        ]

    def visit_EqNode(self, node):
        """
        Converts equation node to sympy equation
        """
        return sym.Eq(self.visit(node.node_a), self.visit(node.node_b), evaluate=False)

    def visit_AssocNode(self, node):
        """
        Converts association node to python dictionary
        """
        try:
            return {
                self.visit(key): self.visit(value)
                for key, value in zip(node.keys, node.values)
            }
        except TypeError:
            return ([self.visit(key) for key in node.keys], [self.visit(value) for value in node.values])

    def visit_GreekNode(self, node):
        """
        Converts greek node to sympy greek letter
        """
        if node.value in greek_letters:
            return sym.Symbol(greek_letters[node.value])
        else:
            return sym.Symbol(node.value)

    def visit_FunctionNode(self, node):
        """
        Converts function node to sympy function
        """
        if node.func == "Derivative":
            if isinstance(node.args[2], list):
                func = sym.Function(self.visit(node.args[1]))(*[sym.sympify(child) for child in node.args[2]])
                derivative = sym.diff(
                    func,
                    *[
                        node.args[2][index]
                        for index, order in enumerate(node.args[0])
                        for _ in range(self.visit(order))
                    ]
                )
            else:
                func = sym.Function(self.visit(node.args[1]))(node.args[2])
                derivative = sym.diff(
                    func,
                    *[
                        node.args[2] for _ in range(self.visit(node.args[0]))
                    ]
                )
            return derivative
        if node.func in known_functions:
            func = known_functions[node.func]
            if isinstance(node.args[0], list):
                return func(*[sym.sympify(child) for child in node.args[0]])
            return func(sym.sympify(node.args[0]))
        if isinstance(node.args[0], list):
            return sym.Function(node.func)(*[sym.sympify(child) for child in node.args[0]])
        return sym.Function(node.func)(sym.sympify(node.args[0]))
