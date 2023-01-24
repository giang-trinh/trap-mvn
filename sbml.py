"""
    # Overview

    This module is responsible for taking an SBML-qual model, reading its update functions and translating
    them into a Petri net encoding that can mirror either the *general* or the *unitary* dynamics of the 
    original system. The translation is performed through BDDs, such that "normal" Boolean variables
    are represented as a single decision variable, while integer variables are expanded into a 1-hot
    encoding (`k` variables for `k` possible values, only one variable can be true at a time).

    Note that there is no intermediate "SBML model" object: the BDDs are constructed directly while the
    function terms are parsed. This is mainly to avoid supporting the whole SBML data model. As such,
    we do not fully check the integrity of the SBML file (we do some basic sanity checks though).

    # Encoding overview

    The encoding is based on "Characterization of Reachable Attractors Using Petri Net Unfoldings"
    (DOI: 10.1007/978-3-319-12982-2_10). Here, an integer variable `X \in [0,k]` is represented by 
    `k` Boolean variables which we denote `pX_b0`, `pX__b1`, ..., `pX_bk`. Subsequently, each Boolean
    variable corresponds to a 1-safe Petri net place. By guaranteeing that there is always *exactly*
    one token between these `k` places, we ensure that they encode a valid integer value.

    A change of value from `p` to `q` then consumes a token in place `pX_bp` and produces a token 
    in place `p{X}__bq`. However, such change can be only available in accordance to the update rules
    of the original SBML model. Furthermore, this can either happen generally (any pair of `p` 
    and `q` is admissible), or in a "unitary" fashion (only +1/-1 changes are possible). 
    
    To generate either general or unitary transitions, we need a complete list of implicants for
    every integer value, such that each implicant is a conjunction of positive Boolean literals in our 
    encoding (e.g. `pX_b0 & pY_b2`). Informally, in the original SBML model, the update 
    function of `x` maps state `s` to value `b` if and only if one of the implicants of `b` 
    is satisfied by `s`. Note that the "only if" part means that no state is allowed to satisfy 
    two implicants of two distinct values.

    # Obtaining value implicants

    To obtain this list of value implicants, for each output value, we construct the BDD
    representing the terms which result in this particular output value. Since an output value 
    can have multiple terms, we have to take a disjunction of these. Finally, we also have to
    deal with the default term by substituting it for a negation of the remaining terms (SBML
    should guarantee that the terms themselves are disjoint, so we don't have to deal with
    that, fortunately). Once we have the implicant BDD for every level of every variable, we 
    also make sure it adheres to the encoding rules (exactly one Boolean variable is true 
    for every encoded integer). Technically, we could do this while creating the BDD, but it
    is not necessary and it is slightly easier to do it after the whole model is parsed.
    
    # Obtaining Petri net transitions from value implicants

    Once we have the implicants for every integer value (of every update function), we can construct
    the Petri net transitions.

    For the general case: We consider every pair of different integer values `a` and `b`. Transitions
    from `a` to `b` are then constructed by considering implicants of `b` restricted to the case 
    where `X=a`. We then create a PN transition which moves a token from `a` to be `b` while 
    checking presence of tokens in the respective implicant places.
    
    For the unitary case: We consider every value `a` with its +1/-1 value `b`. Transitions
    from `a` to `b` are then constructed by considering a union of implicants of every `c >= b`. 
    The transition itself is identical to the general case though.

    Note that in both cases, we now have to "expand" the Boolean variables since these are not in 
    the 1-hot encoding yet. Furthermore, we'd like to create as few transitions as possible.
    Unfortunately, our BDD encoding may create unnecessary implicants for cases where the result
    does not depend on an integer variable, but the variable still appears in the BDD. 
    
    The variable still appears in the BDD due to consistency constraints (exactly one Boolean variable 
    can be true), but we don't have a way to test if it is relevant or not for the actual result. 
    To handle these cases, we have a special expansion which is executed before the enumeration and 
    first isolates values from the BDD which do not depend on the particular integer variable. 
    Technically, the result of this operation is a number of BDDs that is exponential in the worst case. 
    However, the BDDs are disjoint and we would have enumerated their clauses anyway as the next step. 
    So it isn't a problem in practice (any input that fails at this step would have also failed 
    immediately afterwards).

"""

import xml.etree.ElementTree as XmlTree
from xml.etree.ElementTree import Element as XmlElement # type: ignore
from biodivine_aeon import BddVariableSetBuilder, Bdd # type: ignore
import networkx as nx # type: ignore

import sys

# Namespaces of various SBML tags:
NS_SBML = "http://www.sbml.org/sbml/level3/version1/core"
NS_SBML_QUAL = "http://www.sbml.org/sbml/level3/version1/qual/version1"
NS_MATH = "http://www.w3.org/1998/Math/MathML"

# Supported comparison operators in SBML propositions.
PROPOSITION_TESTS = [
    f'{{{NS_MATH}}}eq', 
    f'{{{NS_MATH}}}neq', 
    f'{{{NS_MATH}}}leq', 
    f'{{{NS_MATH}}}geq', 
    f'{{{NS_MATH}}}lt', 
    f'{{{NS_MATH}}}gt'
]


class SymbolicEncoding:
    def __init__(self, levels: dict):
        """
            Create a wrapper object which describes the symbolic encoding of an MVN into Boolean
            BDD variables. The only required argument is a `levels` dictionary which lists 
            the maximum values for all model variables.

            To ensure determinism, the variables are created in the alphabetical order.
        """
        # Ensures deterministic variable order.
        variables = list(sorted(levels.keys()))

        # Dictionary which maps Boolean SBML variables to their BDD counterparts.
        booleans = {}
        # Dictionary which maps integer SBML variables to a list of BDD counterparts.
        integers = {}

        # Set of Boolean BDD variables: used for testing is BDD variable represents a Boolean or an integer.
        boolean_variables = set()

        bdd_vars = BddVariableSetBuilder()
        for var in variables:
            max_level = levels[var]
            if max_level == 1:
                # Boolean variables are expanded later (it keeps the BDDs smaller).
                v = bdd_vars.make_variable(f"p{var}")
                booleans[var] = v
                boolean_variables.add(v)
            else:
                # Integer variables are expanded into `k` distinct Boolean variables immediately.
                integers[var] = bdd_vars.make([f"p{var}_b{x}" for x in range(max_level + 1)])

        self.levels = levels
        self.booleans = booleans
        self.integers = integers
        self.boolean_variables = boolean_variables
        self.bdd_vars = bdd_vars.build()

    def make_symbolic_proposition(self, var: str, op: str, value: int) -> Bdd:
        """
            Build a BDD representing an integer proposition (variable operator value). 

            WARNING: The resulting BDD rejects all properly encoded values that don't match
            the proposition, but it may still admit values which are invalid in the Boolean
            encoding. After you build the whole BDD term, you should run `prune_invalid`
            on the BDD to remove any invalid values.
        """
        assert 0 <= value and value <= self.levels[var], \
            f"Invalid level for variable {var} (max level {self.levels[var]})."

        if var in self.booleans:
            assert op == "eq" or op == "neq", \
                f"Boolean variables only support eq/neq propositions for now."

            positive = False
            if (op == "eq" and value == 1) or (op == "neq" and value == 0):
                positive = True
            return self.bdd_vars.mk_literal(self.booleans[var], positive)
        if var in self.integers:
            max_level = self.levels[var]            
            if op == "eq":
                return self.bdd_vars.mk_literal(self.integers[var][value], True)
            if op == "neq":
                return self.bdd_vars.mk_literal(self.integers[var][value], False)
            if op == "lt":
                # x < 2 <=> x__0 | x__1
                if value == 0:  # x < 0 <=> False
                    return self.bdd_vars.mk_const(False)
                v = { self.integers[var][x]:True for x in range(0, value) }
                return self.bdd_vars.mk_disjunctive_clause(v)
            if op == "gt":
                # x > 2 <=> x__3 | x__4 | ... 
                if value == max_level:  # x > max <=> False
                    return self.bdd_vars.mk_const(False)
                v = { self.integers[var][x]:True for x in range(value + 1, max_level + 1) }
                return self.bdd_vars.mk_disjunctive_clause(v)
            if op == "leq":
                # x <= 2 <=> x__0 | x__1 | x__2
                if value == max_level:  # x <= max <=> True
                    return self.bdd_vars.mk_const(True)
                v = { self.integers[var][x]:True for x in range(0, value + 1) }
                return self.bdd_vars.mk_disjunctive_clause(v)
            if op == "geq":
                # x >= 2 <=> x__2 | x__3 | ...
                if value == 0:  # x >= 0 <=> True
                    return self.bdd_vars.mk_const(True) 
                v = { self.integers[var][x]:True for x in range(value, max_level + 1) }
                return self.bdd_vars.mk_disjunctive_clause(v)
            raise Exception(f"Unknown comparison operator {op}.")
        raise Exception(f"Unknown variable name {var}.")

    def read_mathml_proposition(self, node: XmlElement) -> Bdd:
        """
            Convert an XML tag which represents an integer proposition in MathML into a BDD. 

            Note that the BDD may not reject invalid encodings. You should eventually remove
            invalid values using `prune_invalid`.
        """
        op = node[0].tag    
        assert op in PROPOSITION_TESTS, \
            f"Unknown comparison operator: `{op}`."

        assert len(node) == 3, \
            f"Unexpected number of arguments in a proposition: `{len(node)}`."

        left = read_mathml_literal(node[1])
        right = read_mathml_literal(node[2])
    
        # Strip MathML namespace.
        op = op.replace(f"{{{NS_MATH}}}", "")

        # If the proposition is canonical, we can just build it.
        if isinstance(left, str) and isinstance(right, int):
            return self.make_symbolic_proposition(left, op, right)

        # If the proposition is not canonical, we have to flip it.
        if isinstance(left, int) and isinstance(right, str):
            if op == "eq" or op == "neq":
                op = op
            elif op == "lt":
                op = "gt"
            elif op == "gt":
                op = "lt"
            elif op == "leq":
                op = "geq"
            elif op == "geq":
                op = "leq"

            return self.make_symbolic_proposition(right, op, left)

        # Otherwise, this is an unsupported proposition (e.g. comparing two
        # variables or two Constants). We might add this later, but for now
        # we don't really need it.
        raise Exception(f"Unsupported proposition type {left} {op} {right}.")

    def read_mathml_formula(self, node: XmlElement) -> Bdd:
        """
            Convert an XML tag of a MathML Boolean term with integer propositions
            into a BDD.

            Note that the BDD can still admit incorrectly encoded values. You should
            then use `prune_invalid` to remove these values once you have to.
            (this will often icrease the size of the BDD, hence we delay it as much
            as possible).
        """
        # The tag must be an `apply` node. Anything else is not valid here.
        assert node.tag == f'{{{NS_MATH}}}apply', \
            f"Unsupported mathml expression root: {node.tag}."
    
        op_tag = node[0].tag

        # If the operator is logical, then recursively continue parsing.
        if op_tag == f'{{{NS_MATH}}}not':
            assert len(node) == 2, \
                f"Invalid number of MathML arguments in negation: {len(node)}."
            return self.read_mathml_formula(node[1]).l_not()      
        if op_tag == f'{{{NS_MATH}}}and':
            result = self.bdd_vars.mk_const(True)
            for inner in node[1:]:
                inner = self.read_mathml_formula(inner)
                result = result.l_and(inner)
            return result
        if op_tag == f'{{{NS_MATH}}}or':
            result = self.bdd_vars.mk_const(False)
            for inner in node[1:]:
                inner = self.read_mathml_formula(inner)
                result = result.l_or(inner)
            return result
        if op_tag == f'{{{NS_MATH}}}xor':
            assert len(node) == 3, \
                f"Invalid number of MathML arguments in XOR: {len(node)}."
            left = self.read_mathml_formula(node[1])
            right = self.read_mathml_formula(node[2])
            return left.l_xor(right)
        if op_tag == f'{{{NS_MATH}}}implies':
            assert len(node) == 3, \
                f"Invalid number of MathML arguments in IMPLIES: {len(node)}."
            left = self.read_mathml_formula(node[1])
            right = self.read_mathml_formula(node[2])
            return left.l_imp(right)
        
        # If the operator is a proposition test, continue parsing the XML node as proposition.
        if op_tag in PROPOSITION_TESTS:
            return self.read_mathml_proposition(node)
        
        raise Exception(f"Unsupported mathml apply operator: {op_tag}.")

    def prune_invalid(self, bdd: Bdd, variable: str) -> Bdd:
        """
            Ensure that the BDD only contains properly encoded values of 
            the provided `variable`.
        """
        if variable not in self.integers:
            return bdd
        
        var_levels = self.integers[variable]
        max_level = self.levels[variable]

        # Ensure that at most one encoding variable can be true.
        for l in range(0, max_level + 1):
            # l=1 => (a=0 & b=0 & ...)            
            l_is_true = self.bdd_vars.mk_literal(var_levels[l], True)
            rest = { x:False for x in var_levels if x != var_levels[l] }
            rest_is_false = self.bdd_vars.mk_conjunctive_clause(rest)
            condition = l_is_true.l_imp(rest_is_false)
            bdd = bdd.l_and(condition)
        # Ensure that at least one encoded variable is true.
        # a=1 | b=1 | c=1 | ...
        at_least_one = { x:True for x in var_levels }
        at_least_one = self.bdd_vars.mk_disjunctive_clause(at_least_one) 
        bdd = bdd.l_and(at_least_one)
        return bdd

    def list_implicants(self, bdd: Bdd) -> list[list[str]]:
        """
            Return a list of positivie conjunctive clauses over the encoding variables
            which together imply the validity of this BDD.

            WARNING: Note that the BDD must not contain incorrectly encoded values.
            Use `prune_invalid` first to remove such valuations.
        """
        result = []
        for clause in bdd.list_sat_clauses():
            implicant = []
            for (bdd_var, value) in clause:
                if bdd_var in self.boolean_variables:
                    # Boolean variables have to be expanded.
                    if value:
                        implicant.append(self.bdd_vars.name_of(bdd_var)+"_b1")
                    else:
                        implicant.append(self.bdd_vars.name_of(bdd_var)+"_b0")
                else:
                    # Integer variables have their names baked in, but we 
                    # only take the positive ones.
                    if value:
                        implicant.append(self.bdd_vars.name_of(bdd_var))
            result.append(implicant)
        return result

    def expand_universal_integers(self, bdd: Bdd) -> list[Bdd]:
        """
            Our integer encoding requires that even variables which do not influence
            the final result are included in the BDD, because the incorrectly encoded
            values have to be pruned from the BDD.

            This means enumerating the sat clauses of such BDD will also contain these
            variables even though they are redundant. 

            This operation expands a single BDD into a list of BDDs, such that the
            list of satisfying valuations of the result is the same as for the source
            BDD. However, the BDDs in the result only depend on integer variables
            if the value is actually important.

            This operation can in theory produce many BDDs, but it's never more than
            the number of sat clauses of the original BDD, and as such is preferable
            to listing the sat clauses directly.
        """
        bdds = [bdd]
        for var in self.integers:
            new_bdds = []
            for bdd in bdds:
                shared = self.bdd_vars.mk_const(True)
                for level in range(self.levels[var] + 1):
                    # Variable values which guarantee this particular level.                    
                    restriction = { x: False for x in self.integers[var] }
                    restriction[self.integers[var][level]] = True

                    # Take all values with var=level, remove var, and intersect
                    # with the values that we have so far.
                    shared = shared.l_and(bdd.restrict(restriction))
                if not shared.is_false():
                    # Shared contains valuations which do not depend on var.
                    new_bdds.append(shared)
                
                not_shared = bdd.l_and_not(shared)
                if not not_shared.is_false():
                    # Not shared contains valuations which do depend on var.
                    new_bdds.append(not_shared)
            bdds = new_bdds
        return bdds

def sbml_file_to_petri_net(file_path: str, unitary=False):
    return sbml_to_petri_net_2(XmlTree.parse(file_path), unitary=unitary)

def sbml_string_to_petri_net(xml_string: str, unitary=False):
    return sbml_to_petri_net_2(XmlTree.fromstring(xml_string), unitary=unitary)

def sbml_to_petri_net_2(sbml: XmlElement, unitary=False):
    model = sbml.find(f"{{{NS_SBML}}}model")
    assert model, "Missing <model> tag."

    list_of_species = model.find(f"{{{NS_SBML_QUAL}}}listOfQualitativeSpecies")
    assert list_of_species, "Missing <listOfQualitativeSpecies> tag."

    list_of_transitions = model.find(f"{{{NS_SBML_QUAL}}}listOfTransitions")
    assert list_of_transitions, "Missing <listOfTransitions> tag."

    # Read max. levels of all model variables and create a corresponding symbolic encoding.
    levels = {}
    for var_tag in list_of_species.findall(f"{{{NS_SBML_QUAL}}}qualitativeSpecies"):
        max_level = int(var_tag.attrib[f"{{{NS_SBML_QUAL}}}maxLevel"])
        var_id = str(var_tag.attrib[f"{{{NS_SBML_QUAL}}}id"])
        levels[var_id] = max_level
        
    encoding = SymbolicEncoding(levels)

    # Read implicant BDDs from SBML transitions.
    implicants = {}
    for transition_tag in list_of_transitions.findall(f"{{{NS_SBML_QUAL}}}transition"):
        (output, level_terms) = _read_variable_implicants(encoding, transition_tag)        
        implicants[output] = level_terms
    
    # Now we can start building the Petri net.
    net = nx.DiGraph()

    # Create places for all variables.
    for var in levels:
            for i in range(0, levels[var]+1):
                net.add_node(f"p{var}_b{i}", kind="place")

    tr_id = 0

    # Now add transitions for Boolean variables. These are the same
    # regardless of unitary/general semantics.
    for var in encoding.booleans:
        if var not in implicants:
            continue    # Constant variables might not have transitions.
        bdd_var = encoding.booleans[var]
        p_one = encoding.bdd_vars.name_of(bdd_var) + "_b1"
        p_zero = encoding.bdd_vars.name_of(bdd_var) + "_b0"
        # Consider 0 -> 1 transition.
        # The restriction ensures only relevant implicants are supplied
        # while the modified variable itself does not appear in them.
        bdd = implicants[var][1].var_restrict(bdd_var, False)
        for bdd in encoding.expand_universal_integers(bdd):
            for clause in encoding.list_implicants(bdd):
                tr_id += 1
                _ensure_pn_transition(net, tr_id, p_zero, p_one, clause)
        # Consider 1 -> 0 transition.
        bdd = implicants[var][0].var_restrict(bdd_var, True)
        for bdd in encoding.expand_universal_integers(bdd):
            for clause in encoding.list_implicants(bdd):
                tr_id += 1
                _ensure_pn_transition(net, tr_id, p_one, p_zero, clause)        

    if not unitary:
        # Create general (non-unitary transitions).
        for var in encoding.integers:   
            if var not in implicants:
                continue    # Constant variables might not have transitions.       
            for s_level in range(0, encoding.levels[var] + 1):
                s_bdd_var = encoding.integers[var][s_level]
                s_name = encoding.bdd_vars.name_of(s_bdd_var)
                for t_level in range(0, encoding.levels[var] + 1):
                    if s_level == t_level:
                        continue

                    t_bdd_var = encoding.integers[var][t_level]
                    t_name = encoding.bdd_vars.name_of(t_bdd_var)

                    # Consider s_level -> t_level transition.
                    # Again, we want to eliminate the actual variable from the
                    # implicants while only considering relevant implicants.
                    restriction = { x: False for x in encoding.integers[var] }
                    restriction[s_bdd_var] = True
                    bdd = implicants[var][t_level].restrict(restriction)
                    for bdd in encoding.expand_universal_integers(bdd):
                        for clause in encoding.list_implicants(bdd):
                            tr_id += 1
                            _ensure_pn_transition(net, tr_id, s_name, t_name, clause)                    
    else:
        # Create unitary transitions.
        for var in encoding.integers:
            if var not in implicants:
                continue    # Constant variables might not have transitions.
            for s_level in range(0, encoding.levels[var] + 1):
                s_bdd_var = encoding.integers[var][s_level]
                s_name = encoding.bdd_vars.name_of(s_bdd_var)
                
                # Consider s_level -> s_level + 1 transition.
                if s_level != levels[var]:
                    t_bdd_var = encoding.integers[var][s_level + 1]
                    t_name = encoding.bdd_vars.name_of(t_bdd_var)

                    # Take a union of all implicants for larger levels.
                    bdd = encoding.bdd_vars.mk_const(False)
                    for x in implicants[var][s_level + 1:]:
                        bdd = bdd.l_or(x)

                    # Eliminate the modified variable from the implicants.
                    restriction = { x: False for x in encoding.integers[var] }
                    restriction[s_bdd_var] = True
                    bdd = bdd.restrict(restriction)

                    for bdd in encoding.expand_universal_integers(bdd):
                        for clause in encoding.list_implicants(bdd):
                            tr_id += 1
                            _ensure_pn_transition(net, tr_id, s_name, t_name, clause)
                    
                # Consider s_level -> s_level - 1 transition.
                if s_level > 0:
                    t_bdd_var = encoding.integers[var][s_level - 1]
                    t_name = encoding.bdd_vars.name_of(t_bdd_var)

                    bdd = encoding.bdd_vars.mk_const(False)
                    for x in implicants[var][:s_level]:
                        bdd = bdd.l_or(x)
                
                    restriction = { x: False for x in encoding.integers[var] }
                    restriction[s_bdd_var] = True
                    bdd = bdd.restrict(restriction)

                    for bdd in encoding.expand_universal_integers(bdd):
                        for clause in encoding.list_implicants(bdd):
                            tr_id += 1
                            _ensure_pn_transition(net, tr_id, s_name, t_name, clause)                    

    # print the number of transitions
    #print("# transitions = " + str(tr_id))

    # Finally, trappist needs the levels dictionary with the domain size, not the max level.
    levels = { k: v + 1 for (k, v) in encoding.levels.items() }    
    return (net, levels)

def _ensure_pn_transition(net: nx.DiGraph, tr_id: int, source: str, target: str, conditions: list[str]):
    """
        Create a PN transition moving token from `source` to `target`, requiring that all places in `conditions`
        have a token in them. We assume `conditions` does not contain neither `source` nor `target`.
    """    
    tr_name = f"tr_{tr_id}"
    net.add_node(tr_name, kind="transition")
    net.add_edge(source, tr_name)
    net.add_edge(tr_name, target)   
    for c in conditions:
        net.add_edge(c, tr_name)
        net.add_edge(tr_name, c)    

def _read_variable_implicants(encoding: SymbolicEncoding, transition_tag: XmlElement) -> tuple[str, list[Bdd]]:
    """
        Read the SBML transition tag into a list of implicant BDDs for each output level
        of the corresponding variable.
    """
    transition_id = transition_tag.attrib[f"{{{NS_SBML_QUAL}}}id"]

    list_of_inputs = transition_tag.find(f"{{{NS_SBML_QUAL}}}listOfInputs")
    # Inputs can be empty for source nodes, in which case the list tag is missing entirely.
    list_of_outputs = transition_tag.find(f"{{{NS_SBML_QUAL}}}listOfOutputs")
    # However, an output tag must be present always.
    assert list_of_outputs, \
        f"Missing <listOfOutputs> in transition {transition_id}."

    list_of_terms = transition_tag.find(f"{{{NS_SBML_QUAL}}}listOfFunctionTerms")            
    # We also always expect a list of terms.
    assert list_of_terms, \
        f"Missing <listOfFunctionTerms> in transition {transition_id}."

    # We only support models where each transition has one (unique) output, the effect
    # for that output is assignment, and the effect on inputs is none.
    assert len(list_of_outputs) == 1, \
        f"Invalid number of outputs in transition {transition_id}."
    output = list_of_outputs[0]                        
    assert output.attrib[f"{{{NS_SBML_QUAL}}}transitionEffect"] == "assignmentLevel", \
        f"Unsupported output transition effect in transition {transition_id}."
    
    output = str(output.attrib[f"{{{NS_SBML_QUAL}}}qualitativeSpecies"])
    inputs = []
    if list_of_inputs:
        for input_tag in list_of_inputs:
            assert input_tag.attrib[f"{{{NS_SBML_QUAL}}}transitionEffect"] == "none", \
                f"Unsupported input transition effect in transition {transition_id}."
            inputs.append(str(input_tag.attrib[f"{{{NS_SBML_QUAL}}}qualitativeSpecies"]))

    # There must be a valid default term.
    default_term = list_of_terms.find(f"{{{NS_SBML_QUAL}}}defaultTerm")            
    default_value = int(default_term.attrib[f"{{{NS_SBML_QUAL}}}resultLevel"])
    assert 0 <= default_value and default_value <= encoding.levels[output], \
        f"Invalid default level in transition {transition_id}."
    
    # Prepare list of empty BDDs for every output integer level.
    level_terms = [encoding.bdd_vars.mk_const(False) for i in range(encoding.levels[output] + 1)] 
    
    # Collect all terms of the update function.
    for function_term in list_of_terms.findall(f"{{{NS_SBML_QUAL}}}functionTerm"):
        term_result = int(function_term.attrib[f"{{{NS_SBML_QUAL}}}resultLevel"])
        math = function_term.find(f"{{{NS_MATH}}}math") 
        assert math, \
            f"Missing math element in transition {transition_id}."

        term = encoding.read_mathml_formula(math[0])
        level_terms[term_result] = level_terms[term_result].l_or(term)            
    
    # Add the default term condition.
    default_term_condition = encoding.bdd_vars.mk_const(True)
    for term in level_terms:
        default_term_condition = default_term_condition.l_and_not(term)        
    level_terms[default_value] = level_terms[default_value].l_or(default_term_condition)
    
    # Prune values that are invalid in the 1-hot encoding.
    for var in inputs:
        level_terms = [encoding.prune_invalid(x, var) for x in level_terms]
    
    return (output, level_terms)

def read_mathml_literal(node: XmlElement):    
    """
        Parse a MathML XML node into a literal: Either a string variable name, 
        or an integer value.
    """
    if node.tag == f'{{{NS_MATH}}}ci':
        return node.text.strip()
    elif node.tag == f'{{{NS_MATH}}}cn':
        value_type = node.attrib[f"type"]
        assert value_type == "integer", f"Unsupported MathML value type {value_type}."
        return int(node.text.strip())
    raise Exception(f"Invalid proposition literal: {node}.")

if __name__ == '__main__':
    sbml_file_to_petri_net(sys.argv[1])
