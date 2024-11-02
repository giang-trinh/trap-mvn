from __future__ import annotations
from enum import Enum
import re
import json
import statistics
import math
from pathlib import Path
import xml.etree.ElementTree as ET  # type: ignore
from xml.etree.ElementTree import Element as XmlElement  # type: ignore
from xml.etree.ElementTree import ElementTree as XmlTree  # type: ignore
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List, Set, Tuple, Union, Optional


class ArithOp(Enum):
    ADD = "+"
    MINUS = "-"
    TIMES = "*"
    DIV = "/"

    def __str__(self):
        return self.value


class BMA_Literal:
    literal: Union[str, int]

    def __init__(self, value: Union[str, int]):
        self.literal = value

    def __str__(self):
        return str(self.literal)

    def save_support_set(self, result: Set[str]):
        if type(self.literal) == str:
            result.add(self.literal)

    def evaluate(self, valuation: Dict[str, Tuple[int, Fraction]]) -> Fraction:
        """
        Evaluate this BMA literal in the given variable valuation.
        """
        if type(self.literal) == int:
            return Fraction(numerator=self.literal, denominator=1)
        elif type(self.literal) == str:
            return valuation[self.literal][1]
        else:
            raise Exception("Unreachable.")


class BMA_Expression:
    operator: Union[str, ArithOp]
    arguments: List[Union[BMA_Expression, BMA_Literal]]

    def __init__(
        self,
        operator: Union[str, ArithOp],
        arguments: List[Union[BMA_Expression, BMA_Literal]],
    ):
        self.operator = operator
        self.arguments = arguments

    def __str__(self):
        arg_str = ",".join([str(a) for a in self.arguments])
        return f"`{self.operator}`({arg_str})"

    def save_support_set(self, result: Set[str]):
        for arg in self.arguments:
            arg.save_support_set(result)

    def evaluate(self, valuation: Dict[str, Tuple[int, Fraction]]) -> Fraction:
        """
        Evaluate this BMA expression in the given variable valuation. The first element
        in the valualtion is the nominal integer value. The second is the actual normalized
        fractional value which will be used for computation.
        """
        if type(self.operator) == ArithOp:
            # Evaluate arithmetic expression.
            assert len(self.arguments) == 2
            left = self.arguments[0].evaluate(valuation)
            right = self.arguments[1].evaluate(valuation)
            if self.operator == ArithOp.ADD:
                return left + right
            if self.operator == ArithOp.MINUS:
                return left - right
            if self.operator == ArithOp.TIMES:
                return left * right
            if self.operator == ArithOp.DIV:
                return left / right
        else:
            # Evaluate function expression.
            args = [x.evaluate(valuation) for x in self.arguments]
            if self.operator == "min":
                return min(args)
            if self.operator == "max":
                return max(args)
            if self.operator == "avg":
                return statistics.mean(args)
            if self.operator == "ceil":
                assert (
                    len(args) == 1
                ), f"Ceil function expects one argument, but {len(args)} were found."
                return Fraction(numerator=math.ceil(args[0]))
            if self.operator == "floor":
                assert (
                    len(args) == 1
                ), f"Floor function expects one argument, but {len(args)} were found."
                return Fraction(numerator=math.floor(args[0]))
            if self.operator == "abs":
                assert (
                    len(args) == 1
                ), f"Absolute value expects one argument, but {len(args)} were found."
                return Fraction(numerator=abs(args[0]))
        raise Exception(f"Unknown function operator: {self.operator}.")


class BMA_Function:
    inputs: List[str]
    expression: Union[BMA_Expression, BMA_Literal]

    def __init__(
        self, inputs: List[str], expression: Union[BMA_Expression, BMA_Literal]
    ):
        self.inputs = inputs
        self.expression = expression

    def __str__(self):
        input_str = ",".join(self.inputs)
        return f"fun({input_str}): {str(self.expression)}"

    @staticmethod
    def parse(expression_string: str, variable_names: Dict[int, str]) -> BMA_Function:
        expression_tokens = tokenize_bma_formula(expression_string)
        expression = parse_bma_tokens(expression_tokens, variable_names)
        support: Set[str] = set()
        expression.save_support_set(support)
        inputs = sorted(list(support))
        return BMA_Function(inputs, expression)

    def evaluate(self, valuation: Dict[str, Tuple[int, Fraction]]) -> Fraction:
        """
        Evaluate this BMA expression in the given variable valuation. The first element
        in the valualtion is the nominal integer value. The second is the actual normalized
        fractional value which will be used for computation.
        """
        return self.expression.evaluate(valuation)


class BMA_Model:
    variables: Dict[str, int]
    functions: Dict[str, BMA_Function]

    def __init__(self, variables: Dict[str, int], functions: Dict[str, BMA_Function]):
        self.variables = variables
        self.functions = functions

    @staticmethod
    def from_raw_data(
        variables: List[Tuple[int, str, int, str]],
        regulations: List[Tuple[int, int, str]],
    ) -> BMA_Model:
        """
        Build the BMA model from raw data. This includes a list of variables, each variable described by id, name,
        max level, and formula string. Note that the name can be changed to ensure uniqueness. Also, formula can be
        empty, in which case the default BMA function is used. You also have to supply the list of regulations,
        which consists of source id, target id, and sign (+/-).
        """

        name_to_id = {}
        id_to_name = {}

        zero_constants = []
        max_levels = {}

        # First, normalize variable names if necessary and normalize all
        # zero-constant variables.
        for var in variables:
            v_id = var[0]
            v_name = var[1]
            v_max = var[2]

            # BMA models often contain duplicate variable names or invalid characters.
            v_name = canonical_name(v_name, v_id)

            # Check that the canonical name is unique.
            assert (
                v_name not in name_to_id
            ), f"Duplicate name after normalization: {v_name}."

            name_to_id[v_name] = v_id
            id_to_name[v_id] = v_name

            if v_max == 0:
                # Override zero constants to normal variables.
                # Later as we are building the update functions, we will ensure
                # that the value always tends towards zero.
                zero_constants.append(v_name)
                max_levels[v_name] = 1
            else:
                max_levels[v_name] = v_max

        regulators: Dict[str, List[Tuple[str, str]]] = {x: [] for x in name_to_id}

        # Go through all regulations and group them by target.
        # We will need this information to build the default functions.
        for reg in regulations:
            source = id_to_name[reg[0]]
            target = id_to_name[reg[1]]
            sign = reg[2]
            regulators[target].append((source, sign))

        functions = {}
        # And now we can actually go through all variables again and build
        # the update functions.
        for var in variables:
            v_id = var[0]
            v_name = id_to_name[v_id]
            v_formula = var[3]
            if v_name in zero_constants:
                # Zero constants don't have a formula but can be only zero.
                formula = BMA_Function([v_name], BMA_Literal(0))
            elif len(v_formula) == 0:
                # The formula is empty, which means we have to build a default one
                # the same way as BMA is doing this.
                positive = []
                negative = []
                for regulator, sign in regulators[v_name]:
                    if sign == "+":
                        positive.append(regulator)
                    elif sign == "-":
                        negative.append(regulator)
                if len(positive) == 0 and len(negative) == 0:
                    # This is an undetermined input, in which case we set it to zero,
                    # because that's what BMA does.
                    formula = BMA_Function([v_name], BMA_Literal(0))
                else:
                    n_avr: Union[BMA_Literal, BMA_Expression]
                    p_avr: Union[BMA_Literal, BMA_Expression]
                    if len(negative) == 0:
                        n_avr = BMA_Literal(0)
                    else:
                        n_avr = BMA_Expression(
                            "avg", [BMA_Literal(x) for x in negative]
                        )
                    if len(positive) == 0:
                        # This does not make much sense, because it means any variable with only negative
                        # regulators is ALWAYS a constant zero. But this is how BMA seems to be doing it, so
                        # that's what we are doing as well...
                        p_avr = BMA_Literal(0)
                    else:
                        p_avr = BMA_Expression(
                            "avg", [BMA_Literal(x) for x in positive]
                        )

                    formula = BMA_Function(
                        [x[0] for x in regulators[v_name]],
                        BMA_Expression(ArithOp.MINUS, [p_avr, n_avr]),
                    )
            else:
                formula = BMA_Function.parse(v_formula, id_to_name)

            functions[v_name] = formula

        return BMA_Model(max_levels, functions)

    @staticmethod
    def from_json_str(model: str) -> BMA_Model:
        return BMA_Model.from_json(json.loads(model))

    @staticmethod
    def from_json_file(path: str) -> BMA_Model:
        return BMA_Model.from_json_str(Path(path).read_text())

    @staticmethod
    def from_json(model_data: Any) -> BMA_Model:
        model = model_data["Model"]

        variables = []
        for variable in model["Variables"]:
            v_id = variable["Id"]
            v_name = variable["Name"]

            v_min = int(variable["RangeFrom"])
            v_max = int(variable["RangeTo"])
            assert (
                v_min == 0
            ), f"Only variables with min. level 0 are supported. `{v_name}` has min. level {v_min}."

            v_formula = variable["Formula"]
            variables.append((v_id, v_name, v_max, v_formula))

        regulations = []
        for reg in model["Relationships"]:
            source = int(reg["FromVariable"])
            target = int(reg["ToVariable"])
            sign = "?"
            if reg["Type"] == "Activator":
                sign = "+"
            if reg["Type"] == "Inhibitor":
                sign = "-"
            assert sign != "?", f"Unknown type of regulation: {reg}."
            regulations.append((source, target, sign))

        return BMA_Model.from_raw_data(variables, regulations)

    @staticmethod
    def from_xml_str(model: str) -> BMA_Model:
        return BMA_Model.from_xml(ET.fromstring(model))

    @staticmethod
    def from_xml_file(path: str) -> BMA_Model:
        return BMA_Model.from_xml(ET.parse(path).getroot())

    @staticmethod
    def from_xml(model_data: XmlElement) -> BMA_Model:
        if model_data.tag != "AnalysisInput":
            model = model_data.find("AnalysisInput")
        else:
            model = model_data

        assert model is not None, "Missing `AnalysisInput` tag."
        variables = []
        regulations = []
        for var in model.findall("Variables/Variable"):
            v_id = int(var.attrib["Id"])
            name_tag = var.find("Name")
            min_tag = var.find("RangeFrom")
            max_tag = var.find("RangeTo")
            formula_tag = var.find("Function")
            assert min_tag is not None, f"Missing `RangeFrom` for variable {v_id}."
            assert max_tag is not None, f"Missing `RangeTo` for variable {v_id}."
            assert formula_tag is not None, f"Missing `Function` for variable {v_id}."

            if name_tag is None:
                if "Name" in var.attrib:
                    v_name = var.attrib["Name"]
                else:
                    v_name = f"id_{v_id}"
            else:
                if name_tag.text is None:
                    v_name = f"id_{v_id}"
                else:
                    v_name = name_tag.text

            assert v_name is not None, f"Missing name for variable {v_id}."
            assert min_tag.text is not None, f"Missing min value for variable {v_id}."
            assert max_tag.text is not None, f"Missing max value for variable {v_id}."
            v_min = int(min_tag.text)
            v_max = int(max_tag.text)
            v_formula = formula_tag.text
            if v_formula is None:
                v_formula = ""

            assert (
                v_min == 0
            ), f"Only variables with min. level 0 are supported. `{v_name}` has min. level {v_min}."
            variables.append((v_id, v_name, v_max, v_formula))

        for reg in model.findall("Relationships/Relationship"):
            r_id = int(reg.attrib["Id"])
            source_tag = reg.find("FromVariableId")
            target_tag = reg.find("ToVariableId")
            sign_tag = reg.find("Type")
            assert (
                source_tag is not None and source_tag.text is not None
            ), f"Missing `FromVariableId` for regulation {r_id}."
            assert (
                target_tag is not None and target_tag.text is not None
            ), f"Missing `ToVariableId` for regulation {r_id}."
            assert (
                sign_tag is not None and sign_tag.text is not None
            ), f"Missing `Type` for regulation {r_id}."
            source_id = int(source_tag.text)
            target_id = int(target_tag.text)
            reg_type = sign_tag.text
            if reg_type == "Activator":
                sign = "+"
            elif reg_type == "Inhibitor":
                sign = "-"
            else:
                raise Exception(f"Unknown type of regulation: {reg_type}.")

            regulations.append((source_id, target_id, sign))

        return BMA_Model.from_raw_data(variables, regulations)

    def build_function_table(self, variable: str) -> List[Tuple[Dict[str, int], int]]:
        """
        Build an explicit function table which maps all inputs (name -> level map) to
        the computed function output level.
        """

        def _build_input_list_(
            target_variable: str,
            valuation: Dict[str, Tuple[int, Fraction]],
            remaining_support: List[str],
            levels: Dict[str, int],
        ) -> List[Dict[str, Tuple[int, Fraction]]]:
            """
            Recursively builds the list of all possible function input combinations,
            including their normalized values.
            """
            if len(remaining_support) == 0:
                return [valuation.copy()]

            variable = remaining_support[0]
            remaining_support = remaining_support[1:]
            results: List[Dict[str, Tuple[int, Fraction]]] = []
            for level in range(levels[variable] + 1):
                # The second value is the "normalized level" the authors claim they are using
                # to better align the domains of the two variables.
                # The normalization is slightly simplified since we require all variables to start at 0,
                # hence we don't have to account for different low bounds, only different max bounds.
                valuation[variable] = (
                    level,
                    Fraction(
                        numerator=(level * levels[target_variable]),
                        denominator=levels[variable],
                    ),
                )
                results = results + _build_input_list_(
                    target_variable, valuation, remaining_support, levels
                )
                del valuation[variable]
            return results

        function = self.functions[variable]
        support = function.inputs
        inputs = _build_input_list_(
            variable, {}, function.inputs.copy(), self.variables
        )
        table = []
        for i in inputs:
            result_fraction = function.evaluate(i)
            if int(result_fraction.denominator) == 1:
                # Hopefully, most numbers will not actually be fractions, and we don't have to
                # perform any conversion on these.
                result = int(result_fraction.numerator)
            else:
                # BMA is written in C/C# which performs "round half up" arithemtic. However, Python
                # performs "round half even" airthmetic, which is allegedly better for statistics, but
                # it also means we might return a different value compared to BMA.
                # As such, we have to run it through this magiv formula that will actually perform
                # a proper "round half up" rounding.
                result_decimal = Decimal(result_fraction.numerator) / Decimal(
                    result_fraction.denominator
                )
                result = int(
                    result_decimal.quantize(Decimal(0), rounding=ROUND_HALF_UP)
                )

            # Ensure the result is in the valid range for this variable.
            result = max(0, result)
            result = min(result, self.variables[variable])
            table.append(({x: i[x][0] for x in i}, result))

        return table


class Tokens:
    values: List[Union[int, str, Tokens]]

    def __init__(self, values: List[Union[int, str, Tokens]]):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values.__getitem__(index)

    def __iter__(self):
        return self.values.__iter__()


def tokenize_bma_formula(formula: str) -> Tokens:
    """
    Creates a token tree from a BMA formula. A token is a number, operator or a name.
    A subtree is recursively created whenever paretheses are encountered. The tree
    is represented simply a list of lists: If the item is a token, its a string. If
    it is a subtree, it is another list.
    """

    def _tokenize_(formula: str, index: int, top=False) -> Tuple[Tokens, int]:
        """
        Top indicates that this is the top level formula and as such shouldn't
        end with a parenthesis. Otherwise, all other formulas must end with
        parenthesis.
        """
        tokens: List[Union[int, str, Tokens]] = []
        while index < len(formula):
            # Handle single-char tokens:
            if (
                formula[index] == " "
                or formula[index] == "\t"
                or formula[index] == "\n"
            ):
                # Skip whitespace.
                index += 1
            elif formula[index] == "(":
                # Open parenthesis... process recursively and continue.
                (inner_tokens, updated_index) = _tokenize_(formula, index + 1)
                assert updated_index > index
                index = updated_index
                tokens.append(inner_tokens)
            elif formula[index] == ")":
                # Closing parenthesis... if this is the "top" execution, this is
                # an error. Otherwise it means we are done parsing this "level".
                if top:
                    raise Exception(
                        f"Unexpected `)` at position {index} in `{formula}`."
                    )
                else:
                    return (Tokens(tokens), index + 1)
            elif formula[index] == ",":
                # Commas can separate function arguments, but are just copied.
                tokens.append(str(formula[index]))
                index += 1
            elif (
                formula[index] == "*"
                or formula[index] == "+"
                or formula[index] == "-"
                or formula[index] == "/"
            ):
                # Arithmetic operators are also just tokens.
                tokens.append(str(formula[index]))
                index += 1
            else:
                number_match = re.search("^\\d+", formula[index:])
                if number_match:
                    index_after_last = number_match.span()[1]
                    tokens.append(int(number_match.group()))
                    index += index_after_last
                    continue

                name_match = re.search("^[a-zA-Z_][a-zA-Z0-9_]*", formula[index:])
                if name_match:
                    index_after_last = name_match.span()[1]
                    tokens.append(str(name_match.group()))
                    index += index_after_last
                    continue

                raise Exception(f"Unknown tokens at position {index} in `{formula}`.")

        if top:
            # If we have nothing else to read and this is the "top" execution,
            # this is expected and we can just return the result.
            return (Tokens(tokens), index)
        else:
            # However, in any other place this is an error.
            raise Exception(f"Missing `)` in formula `{formula}`.")

    (tokens, index) = _tokenize_(formula, 0, top=True)
    assert index == len(formula), f"Unexpected trailing tokens: {formula[index:]}"
    return tokens


def parse_bma_tokens(tokens: Tokens, names: Dict[int, str]):
    """
    This function parses the token tree produced by `tokenize_bma_formula`
    into a proper object-based representation of the formula. Every formula
    is one of the following objects:
     - (a) an integer literal
     - (b) a string variable name (resolved through `ids`)
     - (c) an application of a binary arithemtic operator: `{ 'op': .., 'left': .., 'right': .. }`.
     - (d) an application of a function to some number of arguments: `{ 'fn': .., 'args': [ .. ] }`.
    """

    # The parsing is performed recursively, such that each function is responsible for handling one
    # type of arithmetic operator. The hierarchy of funcitons then implements operator priority.

    def _last_token_at_(tokens: Tokens, find: List[Union[str, int]]) -> Optional[int]:
        """
        Find the last occurence of any token in the `find` list.
        If no such token is found, returns `None`.
        """
        for i in range(len(tokens)):
            if tokens[-i - 1] in find:
                return len(tokens) - i - 1
        return None

    def _split_by_token_(tokens: Tokens, delimiter: Union[str, int]) -> List[Tokens]:
        """
        Split the given list of tokens into a list of lists of tokens using
        the given delimiter token.
        """
        result = []
        sublist: List[Union[int, str, Tokens]] = []
        for token in tokens:
            if token == delimiter:
                result.append(Tokens(sublist))
                sublist = []
            else:
                sublist.append(token)
        result.append(Tokens(sublist))
        return result

    def _parse_literal_(
        tokens: Tokens, names: Dict[int, str]
    ) -> Union[BMA_Expression, BMA_Literal]:
        # There are four options for a literal:
        if len(tokens) == 0:
            raise Exception(f"Unexpected empty literal.")
        if len(tokens) == 1 and type(tokens[0]) == Tokens:
            # This is a "redundant" parenthesis block, which is just transparently
            # forwarded to a new parser call.
            return _parse_tokens_(tokens[0], names)
        if len(tokens) == 1 and type(tokens[0]) == int:
            # Create an integer literal.
            return BMA_Literal(tokens[0])
        if len(tokens) == 2 and tokens[0] == "var" and type(tokens[1]) == Tokens:
            # Variable literals must be followed by exactly one subtree
            # with one integer ID inside. Anything else is an error.
            if len(tokens[1]) == 1 and type(tokens[1][0]) == int:
                if tokens[1][0] not in names:
                    raise Exception(f"Unknown variable id `{tokens[1][0]}`.")
                return BMA_Literal(names[tokens[1][0]])
            else:
                raise Exception(f"Malformed tokens for variable literal: {tokens}.")
        if len(tokens) == 2 and type(tokens[0]) == str and type(tokens[1]) == Tokens:
            # Function calls are similar, but they don't start with "var", but rather
            # an arbitrary name.
            f_name = tokens[0]
            # Split the arguments by comma.
            f_args = _split_by_token_(tokens[1], ",")
            # And parse each argument separately.
            f_arg_expr = [_parse_tokens_(x, names) for x in f_args]
            return BMA_Expression(operator=f_name, arguments=f_arg_expr)

        # If we got here, it means the tokens did not pass any criteria for any literal type.
        raise Exception(f"Malformed literal: {tokens}.")

    def _parse_multiplication_(
        tokens: Tokens, names: Dict[int, str]
    ) -> Union[BMA_Expression, BMA_Literal]:
        op_index = _last_token_at_(tokens, ["*", "/"])
        if op_index is None:
            # There is no multiplication, we can continue straight to addition.
            return _parse_literal_(tokens, names)
        else:
            # Split based on op_index and recursively continue in both halves.
            if tokens[op_index] == "*":
                operator = ArithOp.TIMES
            if tokens[op_index] == "/":
                operator = ArithOp.DIV

            return BMA_Expression(
                operator,
                arguments=[
                    _parse_multiplication_(Tokens(tokens.values[:op_index]), names),
                    _parse_literal_(Tokens(tokens.values[op_index + 1 :]), names),
                ],
            )

    def _parse_addition_(
        tokens: Tokens, names: Dict[int, str]
    ) -> Union[BMA_Expression, BMA_Literal]:
        op_index = _last_token_at_(tokens, ["+", "-"])
        if op_index is None:
            # There is not addition, continue straits to literals.
            return _parse_multiplication_(tokens, names)
        else:
            # Split based on op_index and recursively continue.
            if tokens[op_index] == "+":
                operator = ArithOp.ADD
            if tokens[op_index] == "-":
                operator = ArithOp.MINUS

            return BMA_Expression(
                operator,
                arguments=[
                    _parse_addition_(Tokens(tokens.values[:op_index]), names),
                    _parse_multiplication_(
                        Tokens(tokens.values[op_index + 1 :]), names
                    ),
                ],
            )

    def _parse_tokens_(
        tokens: Tokens, names: Dict[int, str]
    ) -> Union[BMA_Expression, BMA_Literal]:
        return _parse_addition_(tokens, names)

    return _parse_tokens_(tokens, names)


def canonical_name(v_name, v_id):
    """
    Compute a canonical name which will be unique within the BMA model and not contain any
    special characters (as opposed to BMA, which allows both special characters and duplicate
    names).
    """
    v_name = re.sub("[^0-9a-zA-Z_]", "_", v_name)
    return f"{v_name}_id{str(v_id)}"
