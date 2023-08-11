from __future__ import annotations
from enum import Enum

import xml.etree.ElementTree as ET # type: ignore
from xml.etree.ElementTree import Element as XmlElement # type: ignore
from xml.etree.ElementTree import ElementTree as XmlTree # type: ignore
from biodivine_aeon import BddVariable # type: ignore

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, List, Union, Optional
    from biodivine_aeon import Bdd, BddVariableSet # type: ignore

NS_SBML = "http://www.sbml.org/sbml/level3/version1/core"
NS_SBML_QUAL = "http://www.sbml.org/sbml/level3/version1/qual/version1"
NS_MATH = "http://www.w3.org/1998/Math/MathML"

def literal_from_xml(node: XmlElement) -> Union[int, str]:    
    """
        Parse a MathML XML node into a literal: Either a string name or an integer value.
    """

    if node.text is None:
        raise Exception("Missing text value in literal.")
        
    if node.tag == f'{{{NS_MATH}}}ci':        
        return node.text.strip()
    elif node.tag == f'{{{NS_MATH}}}cn':
        value_type = node.attrib[f"type"]
        assert value_type == "integer", f"Unsupported MathML value type {value_type}."
        return int(node.text.strip())
    raise Exception(f"Unsupported MathML literal: {node}.")

def literal_to_xml(literal: Union[int, str], include_namespace=True) -> XmlElement:
    """
        Convert a "literal" value (integer or string) into MathML XML element.
    """
    if include_namespace:
        if type(literal) == str:
            e = XmlElement(f"{{{NS_MATH}}}ci")        
        elif type(literal) == int:
            e = XmlElement(f"{{{NS_MATH}}}cn", type="integer")
        else:
            raise Exception(f"Unknown MathML literal: {literal}.")
    else:
        if type(literal) == str:
            e = XmlElement(f"ci")        
        elif type(literal) == int:
            e = XmlElement(f"cn", type="integer")
        else:
            raise Exception(f"Unknown MathML literal: {literal}.")
    e.text = str(literal)
    return e

class LogicOp(Enum):
    NOT = "not"
    AND = "and"
    OR = "or"
    XOR = "xor"
    IMPLIES = "implies"

    @staticmethod
    def try_from_xml(element: XmlElement) -> Optional[LogicOp]:
        key = element.tag.replace(f"{{{NS_MATH}}}", "").upper()
        try: 
            return LogicOp[key]        
        except:
            return None
        
    def to_xml(self, include_namespace=True) -> XmlElement:
        if include_namespace:
            return XmlElement(f"{{{NS_MATH}}}{self.value}")          
        else:
            return XmlElement(self.value)


class CmpOp(Enum):
    EQ = "eq",
    NEQ = "neq",
    LT = "lt",
    LEQ = "leq",
    GT = "gt",
    GEQ = "geq"

    def flip(self) -> CmpOp:
        """
            Flip this comparison operator (exchange meaning of left-right operands).
        """
        if self == CmpOp.LT:
            return CmpOp.GT
        if self == CmpOp.GT:
            return CmpOp.LT
        if self == CmpOp.GEQ:
            return CmpOp.LEQ
        if self == CmpOp.LEQ:
            return CmpOp.GEQ 
        
        return self

    @staticmethod
    def from_xml(element: XmlElement) -> CmpOp:
        return CmpOp[element.tag.replace(f"{{{NS_MATH}}}", "").upper()]

    def to_xml(self, include_namespace=True) -> XmlElement:
        # For some weird reason, some of the operators are printed in brackets and some are not
        # if we just use self.value.
        if self == CmpOp.EQ:
            tag = "eq"
        if self == CmpOp.NEQ:
            tag = "neq"
        if self == CmpOp.LT:
            tag = "lt"
        if self == CmpOp.LEQ:
            tag = "leq"
        if self == CmpOp.GT:
            tag = "gt"
        if self == CmpOp.GEQ:
            tag = "geq"
        if include_namespace:
            return XmlElement(f"{{{NS_MATH}}}{tag}")          
        else:
            return XmlElement(tag)

class SBML_Proposition:
    """
        A canonical representation of a proposition in SBML. While technically, this format does not cover
        all possible propositions that can be expressed in SBML, this should be enough to cover everything
        that commonly appears in logical models.

        The proposition consists of a `variable`, `operator`, and a `constant`. For example, when `variable=x`,
        `operator=eq` and `constant=3`, the proposition is interpreted as `x == 3`.
    """
    variable: str
    operator: CmpOp
    constant: int

    def __init__(self, variable: str, operator: CmpOp, constant: int):
        self.variable = variable
        self.operator = operator
        self.constant = constant

    @staticmethod
    def from_xml(element: XmlElement) -> SBML_Proposition:
        """
            Read a canonical SBML proposition from a MathML XML `<apply>` tag.
        """
        # First child must be the comparison operator.         
        op = CmpOp.from_xml(element[0])

        # There must be two other child nodes, both literals.
        assert len(element) == 3, \
            f"Unexpected number of arguments in a proposition: `{len(element)}`."

        left = literal_from_xml(element[1])
        right = literal_from_xml(element[2])

        # If the proposition is canonical, we can just build it, otherwise
        # we have to flip the comparison operator.
        if isinstance(left, str) and isinstance(right, int):
            return SBML_Proposition(left, op, right)
        elif isinstance(left, int) and isinstance(right, str):
            return SBML_Proposition(right, op.flip(), left)
        else:
            # Otherwise, this is an unsupported proposition (e.g. comparing two
            # variables or two Constants). We might add this later, but for now
            # we don't really need it.
            raise Exception(f"Unsupported proposition type {left} {op} {right}.")        

    def to_xml(self, include_namespace=True) -> XmlElement:
        if include_namespace:
            e = XmlElement(f"{{{NS_MATH}}}apply")
        else:
            e = XmlElement("apply")
        e.append(self.operator.to_xml(include_namespace))
        e.append(literal_to_xml(self.variable, include_namespace))
        e.append(literal_to_xml(self.constant, include_namespace))
        return e

    def evaluate_symbolic(self, ctx: BddVariableSet, symbolic: Union[BddVariable, List[BddVariable]]) -> Bdd:
        """
            Build a BDD representing this integer proposition. The argument is either a 
            single `BddVariable`, if this proposition uses a Boolean variable, or a list 
            of `BddVariables` if this proposition concerns an integer variable.

            WARNING: The resulting BDD rejects all properly encoded values that don't match
            the proposition, but it may still admit values which are invalid in the Boolean
            encoding. After you build the whole BDD term, you should run `prune_invalid`
            on the BDD to remove any invalid values.
        """
        if type(symbolic) == BddVariable:
            assert self.operator == CmpOp.EQ or self.operator == CmpOp.NEQ, \
                f"Boolean variables only support eq/neq propositions for now."

            positive = False
            if (self.operator == CmpOp.EQ and self.constant == 1) or (self.operator == CmpOp.NEQ and self.constant == 0):
                positive = True
            return ctx.mk_literal(symbolic, positive)
        if type(symbolic) == list:
            max_level = len(symbolic) - 1
            assert 0 <= self.constant and self.constant <= max_level, \
                f"Invalid level {self.constant} for variable {self.variable} (max level {max_level})."

            if self.operator == CmpOp.EQ:
                return ctx.mk_literal(symbolic[self.constant], True)
            if self.operator == CmpOp.NEQ:
                return ctx.mk_literal(symbolic[self.constant], False)
            if self.operator == CmpOp.LT:
                # x < 2 <=> x__0 | x__1
                if self.constant == 0:  # x < 0 <=> False
                    return ctx.mk_const(False)
                v = { symbolic[x]:True for x in range(0, self.constant) }
                return ctx.mk_disjunctive_clause(v)
            if self.operator == CmpOp.GT:
                # x > 2 <=> x__3 | x__4 | ... 
                if self.constant == max_level:  # x > max <=> False
                    return ctx.mk_const(False)
                v = { symbolic[x]:True for x in range(self.constant + 1, max_level + 1) }
                return ctx.mk_disjunctive_clause(v)
            if self.operator == CmpOp.LEQ:
                # x <= 2 <=> x__0 | x__1 | x__2
                if self.constant == max_level:  # x <= max <=> True
                    return ctx.mk_const(True)
                v = { symbolic[x]:True for x in range(0, self.constant + 1) }
                return ctx.mk_disjunctive_clause(v)
            if self.operator == CmpOp.GEQ:
                # x >= 2 <=> x__2 | x__3 | ...
                if self.constant == 0:  # x >= 0 <=> True
                    return ctx.mk_const(True) 
                v = { symbolic[x]:True for x in range(self.constant, max_level + 1) }
                return ctx.mk_disjunctive_clause(v)
            raise Exception(f"Unknown comparison operator {self.operator}.")            
        raise Exception(f"Unreachable.")

class SBML_Expression:
    """
        A canonical representation of an SBML function expression that is not a proposition.
    """
    operator: LogicOp
    arguments: List[Union[SBML_Expression, SBML_Proposition]]

    def __init__(self, operator: LogicOp, arguments: List[Union[SBML_Expression, SBML_Proposition]]):
        self.operator = operator
        self.arguments = arguments

    @staticmethod
    def from_xml(element: XmlElement) -> Union[SBML_Expression, SBML_Proposition]:
        """
            Read an SBML expression from a MathML XML element. The result is either a `SBML_Expression`
            if the tag represents a complex expression, or `SBML_Proposition` if the tag is only a proposition.
        """

        # The tag must be an `apply` node. Anything else is not valid here.
        assert element.tag == f'{{{NS_MATH}}}apply', \
            f"Unsupported mathml expression root: {element.tag}."
    
        operator = LogicOp.try_from_xml(element[0])
        
        if operator is None:
            # This is not a logic operator, hence it must be a proposition.
            return SBML_Proposition.from_xml(element)
        
        # If the operator is a logic operator, check the number of arguments 
        # and continue parsing recursively.

        if operator == LogicOp.NOT:
            assert len(element) == 2, \
                f"Invalid number of MathML arguments in negation: {len(element) - 1}."

        if operator == LogicOp.XOR or operator == LogicOp.IMPLIES:
            assert len(element) == 3, \
                f"Invalid number of MathML arguments in {operator.value}: {len(element) - 1}."
            
        arguments = [ SBML_Expression.from_xml(e) for e in element[1:] ]
        return SBML_Expression(operator, arguments)

    def to_xml(self, include_namespace=True) -> XmlElement:
        if include_namespace:            
            e = XmlElement(f"{{{NS_MATH}}}apply")
        else:
            e = XmlElement("apply")
        e.append(self.operator.to_xml(include_namespace))
        for arg in self.arguments:
            e.append(arg.to_xml(include_namespace))
        return e

    def evaluate_symbolic(self, ctx: BddVariableSet, booleans: Dict[str, BddVariable], integers: Dict[str, List[BddVariable]]) -> Bdd:
        """
            Build a BDD representing this expression. The arguments map model variables to their
            conterparts in the symbolic encoding.

            WARNING: The resulting BDD rejects all properly encoded values that don't match
            the proposition, but it may still admit values which are invalid in the Boolean
            encoding. After you build the whole BDD term, you should run `prune_invalid`
            on the BDD to remove any invalid values.
        """
        args = []
        for arg in self.arguments:
            if type(arg) == SBML_Proposition:
                if arg.variable in booleans:
                    args.append(arg.evaluate_symbolic(ctx, booleans[arg.variable]))
                else:
                    args.append(arg.evaluate_symbolic(ctx, integers[arg.variable]))
            if type(arg) == SBML_Expression:
                args.append(arg.evaluate_symbolic(ctx, booleans, integers))

        if self.operator == LogicOp.NOT:
            return args[0].l_not()
        if self.operator == LogicOp.AND:
            result = ctx.mk_const(True)
            for arg in args:
                result = result.l_and(arg)
            return result
        if self.operator == LogicOp.OR:
            result = ctx.mk_const(False)
            for arg in args:
                result = result.l_or(arg)
            return result
        if self.operator == LogicOp.IMPLIES:
            return args[0].l_imp(args[1])
        if self.operator == LogicOp.XOR:
            return args[0].l_xor(args[1])
        raise Exception("Unreachable")

class SBML_Term:
    """
        Representation of an SBML term which consists of a integer `result`,
        and an SBML expression.
    """
    result: int
    expression: Union[SBML_Expression, SBML_Proposition]

    def __init__(self, result: int, expression: Union[SBML_Expression, SBML_Proposition]):
        self.result = result
        self.expression = expression

    @staticmethod
    def from_xml(element: XmlElement) -> SBML_Term:
        result = int(element.attrib[f"{{{NS_SBML_QUAL}}}resultLevel"])
        math = element.find(f"{{{NS_MATH}}}math") 
        assert math, "Missing math element in transition term."

        expression = SBML_Expression.from_xml(math[0])
        return SBML_Term(result, expression)

    def to_xml(self) -> XmlElement:        
        e = XmlElement(f"{{{NS_SBML_QUAL}}}functionTerm", attrib={ f"{{{NS_SBML_QUAL}}}resultLevel": str(self.result) })
        # MathML namespace is forced as "default namespace" for this subtree,
        # hence we can set `include_namespace=False` when generating the expressions.
        math = XmlElement(
            f"math",
            attrib={ "xmlns": NS_MATH }
        )
        math.append(self.expression.to_xml(include_namespace=False))
        e.append(math)
        return e

    def evaluate_symbolic(self, ctx: BddVariableSet, booleans: Dict[str, BddVariable], integers: Dict[str, List[BddVariable]]) -> Bdd:
        """
            Build a BDD representing this SBML term. The arguments map model variables to their
            conterparts in the symbolic encoding.

            WARNING: The resulting BDD rejects all properly encoded values that don't match
            the proposition, but it may still admit values which are invalid in the Boolean
            encoding. After you build the whole BDD term, you should run `prune_invalid`
            on the BDD to remove any invalid values.
        """
        if type(self.expression) == SBML_Proposition:
            prop = self.expression
            if prop.variable in booleans:
                return prop.evaluate_symbolic(ctx, booleans[prop.variable])
            else:
                return prop.evaluate_symbolic(ctx, integers[prop.variable])
        elif type(self.expression) == SBML_Expression:
            return self.expression.evaluate_symbolic(ctx, booleans, integers)
        raise Exception("Unreachable.")
    

class SBML_Function:
    inputs: List[str]
    output: str
    default_result: int
    terms: List[SBML_Term]
    def __init__(self, inputs: List[str], output: str, default_result: int, terms: List[SBML_Term]):
        self.inputs = inputs
        self.output = output
        self.default_result = default_result
        self.terms = terms
    
    @staticmethod
    def from_xml(element: XmlElement) -> SBML_Function:
        """
            Parse the SBML transition tag into a proper SBML function object.        
        """
        transition_id = element.attrib[f"{{{NS_SBML_QUAL}}}id"]

        list_of_inputs = element.find(f"{{{NS_SBML_QUAL}}}listOfInputs")
        # Inputs can be empty for source nodes, in which case the list tag is missing entirely.
        list_of_outputs = element.find(f"{{{NS_SBML_QUAL}}}listOfOutputs")
        # However, an output tag must be present always.
        assert list_of_outputs, \
            f"Missing <listOfOutputs> in transition {transition_id}."

        list_of_terms = element.find(f"{{{NS_SBML_QUAL}}}listOfFunctionTerms")            
        # We also always expect a list of terms.
        assert list_of_terms, \
            f"Missing <listOfFunctionTerms> in transition {transition_id}."

        # We only support models where each transition has one (unique) output, the effect
        # for that output is assignment, and the effect on inputs is none.
        assert len(list_of_outputs) == 1, \
            f"Invalid number of outputs in transition {transition_id}."
        output_tag = list_of_outputs[0]                        
        assert output_tag.attrib[f"{{{NS_SBML_QUAL}}}transitionEffect"] == "assignmentLevel", \
            f"Unsupported output transition effect in transition {transition_id}."
    
        output = str(output_tag.attrib[f"{{{NS_SBML_QUAL}}}qualitativeSpecies"])
        inputs = []
        if list_of_inputs:
            for input_tag in list_of_inputs:
                assert input_tag.attrib[f"{{{NS_SBML_QUAL}}}transitionEffect"] == "none", \
                    f"Unsupported input transition effect in transition {transition_id}."
                inputs.append(str(input_tag.attrib[f"{{{NS_SBML_QUAL}}}qualitativeSpecies"]))

        # There must be a valid default term.
        default_term = list_of_terms.find(f"{{{NS_SBML_QUAL}}}defaultTerm")
        assert default_term is not None, \
            f"Missing defult term in transition {transition_id}"

        default_result = int(default_term.attrib[f"{{{NS_SBML_QUAL}}}resultLevel"])        
    
        terms = [SBML_Term.from_xml(term) for term in list_of_terms.findall(f"{{{NS_SBML_QUAL}}}functionTerm")]

        return SBML_Function(inputs, output, default_result, terms)  
    
    def to_xml(self) -> XmlElement:
        list_of_inputs = XmlElement(f"{{{NS_SBML_QUAL}}}listOfInputs")
        for i in self.inputs:
            list_of_inputs.append(XmlElement(
                f"{{{NS_SBML_QUAL}}}input",
                attrib= {
                    f"{{{NS_SBML_QUAL}}}qualitativeSpecies": i,
                    f"{{{NS_SBML_QUAL}}}transitionEffect": "none",
                    f"{{{NS_SBML_QUAL}}}sign": "unknown",
                    f"{{{NS_SBML_QUAL}}}id": f"tr_{self.output}_in_{i}"
                }
            ))

        list_of_outputs = XmlElement(f"{{{NS_SBML_QUAL}}}listOfOutputs")
        list_of_outputs.append(XmlElement(
            f"{{{NS_SBML_QUAL}}}output",
            attrib={
                f"{{{NS_SBML_QUAL}}}qualitativeSpecies": self.output,
                f"{{{NS_SBML_QUAL}}}transitionEffect": "assignmentLevel",
                f"{{{NS_SBML_QUAL}}}id": f"tr_{self.output}_out"
            }
        ))

        list_of_terms = XmlElement(f"{{{NS_SBML_QUAL}}}listOfFunctionTerms")
        list_of_terms.append(XmlElement(
            f"{{{NS_SBML_QUAL}}}defaultTerm",
            attrib={ f"{{{NS_SBML_QUAL}}}resultLevel": str(self.default_result) }
        ))
        
        for term in self.terms:
            list_of_terms.append(term.to_xml())

        transition = XmlElement(f"{{{NS_SBML_QUAL}}}transition", attrib={ f"{{{NS_SBML_QUAL}}}id": f"tr_{self.output}" })
        transition.append(list_of_inputs)
        transition.append(list_of_outputs)
        transition.append(list_of_terms)
        return transition

class SBML_Model:
    variables: Dict[str, int]
    functions: Dict[str, SBML_Function]

    def __init__(self, variables: Dict[str, int], functions: Dict[str, SBML_Function]):
        self.variables = variables
        self.functions = functions

    @staticmethod
    def from_xml(tree: XmlElement) -> SBML_Model:
        model = tree.find(f"{{{NS_SBML}}}model")
        assert model, "Missing <model> tag."

        list_of_species = model.find(f"{{{NS_SBML_QUAL}}}listOfQualitativeSpecies")
        assert list_of_species, "Missing <listOfQualitativeSpecies> tag."

        list_of_transitions = model.find(f"{{{NS_SBML_QUAL}}}listOfTransitions")
        assert list_of_transitions, "Missing <listOfTransitions> tag."

        # Read max. levels of all model variables and create a corresponding entry.
        variables = {}
        for var_tag in list_of_species.findall(f"{{{NS_SBML_QUAL}}}qualitativeSpecies"):
            max_level = int(var_tag.attrib[f"{{{NS_SBML_QUAL}}}maxLevel"])
            var_id = str(var_tag.attrib[f"{{{NS_SBML_QUAL}}}id"])
            variables[var_id] = max_level
        
        functions = {}
        for transition_tag in list_of_transitions.findall(f"{{{NS_SBML_QUAL}}}transition"):
            function = SBML_Function.from_xml(transition_tag)
            functions[function.output] = function
        
        return SBML_Model(variables, functions)

    @staticmethod
    def from_string(model: str) -> SBML_Model:
        return SBML_Model.from_xml(ET.fromstring(model))

    @staticmethod
    def from_file(path: str) -> SBML_Model:
        return SBML_Model.from_xml(ET.parse(path).getroot())

    def to_xml_string(self) -> str:
        return ET.tostring(self.to_xml()).decode()

    def to_file(self, path: str):
        XmlTree(self.to_xml()).write(path, encoding="utf-8")

    def to_xml(self) -> XmlElement:
        ET.register_namespace("sbml", NS_SBML)
        ET.register_namespace("qual", NS_SBML_QUAL)
        ET.register_namespace("math", NS_MATH)


        list_of_species = XmlElement(f"{{{NS_SBML_QUAL}}}listOfQualitativeSpecies")
        for variable in self.variables:
            max_level = self.variables[variable]
            list_of_species.append(XmlElement(
                f"{{{NS_SBML_QUAL}}}qualitativeSpecies",
                attrib={
                    f"{{{NS_SBML_QUAL}}}maxLevel": str(max_level),
                    f"{{{NS_SBML_QUAL}}}id": variable,
                    f"{{{NS_SBML_QUAL}}}name": variable,
                    f"{{{NS_SBML_QUAL}}}compartment": "comp1"
                }
            ))
        
        list_of_transitions = XmlElement(f"{{{NS_SBML_QUAL}}}listOfTransitions")
        for variable in self.functions:
            function = self.functions[variable]
            list_of_transitions.append(function.to_xml())

        list_of_compartments = XmlElement("listOfCompartments")
        list_of_compartments.append(XmlElement("compartment", attrib={ "id": "comp1" }))

        model = XmlElement(f"model", attrib={ "id": "model_id" })
        model.append(list_of_species)
        model.append(list_of_transitions)
        model.append(list_of_compartments)

        sbml = XmlElement(f"sbml", attrib={ 
            "xmlns": NS_SBML,
            f"{{{NS_SBML_QUAL}}}required": "true",
            "level": "3",
            "version": "1"
        })
        sbml.append(model)
        return sbml