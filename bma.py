"""
    This module allows us to read BMA (bio-models analyzer) files. These files are essentially `.json` containing 
    the list of variables together with update functions. The only problem is that the update functions are not
    logical, but rather arithmetic (`*`, `min`, `avg`, ...). However, the output of these functions are still integers.

    As an example, see the `real-world-logical-models/COVID_model.json` file.

    For now, our strategy is to just "brute froce" the translation of these functions, building an input-output table
    and then converting it to a BDD from which we can generate a Petri net.
"""

import json
import sys
import re
import statistics
import math
import networkx as nx
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, ROUND_HALF_UP
from biodivine_aeon import BddVariableSetBuilder, Bdd

class SymbolicEncoding:
    def __init__(self, levels: dict):
        """
            Create a wrapper object which describes the symbolic encoding of an BMA model into Boolean
            BDD variables. The only required argument is a `levels` dictionary which lists 
            the maximum values for all model variables.

            To ensure determinism, the variables are created in the alphabetical order.

            Note: This is shares basic principles with the `SymbolicEncoding` in the `sbml` module, 
            but has various different methods applicable to the BMA syntax instead of MathML. At some 
            point, we might want to merge the two, but for now I'm keeping them separate.
        """
        # Ensures deterministic variable order.
        variables = list(sorted(levels.keys()))

        # Dictionary which maps Boolean BMA variables to their BDD counterparts.
        booleans = {}
        # Dictionary which maps integer BMA variables to a list of BDD counterparts.
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

    def build_implicant_bdds(self, variable, function_table):
        implicants = [self.bdd_vars.mk_const(False) for x in range(self.levels[variable] + 1)]
        for (valuation, output) in function_table:
            clause = {}
            for var in valuation:
                level = valuation[var][0]
                if var in self.booleans:
                    clause[self.booleans[var]] = level == 1
                else:
                    for i in range(self.levels[var] + 1):
                        level_var = self.integers[var][i]
                        clause[level_var] = (i == level)
            clause_bdd = self.bdd_vars.mk_conjunctive_clause(clause)
            implicants[output] = implicants[output].l_or(clause_bdd)

        # We don't have to prune invalid valuations because every clause represents
        # only valid encodings and we are doing a union of clauses.
        return implicants 

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



    
def tokenize_bma_formula(formula):
    """
        Creates a token tree from a BMA formula. A token is a number, operator or a name.
        A subtree is recursively created whenever paretheses are encountered. The tree
        is represented simply a list of lists: If the item is a token, its a string. If 
        it is a subtree, it is another list.
    """
    def _tokenize_(formula, index, top=False):
        """
            Top indicates that this is the top level formula and as such shouldn't
            end with a parenthesis. Otherwise, all other formulas must end with 
            parenthesis.
        """
        tokens = []
        while index < len(formula):
            # Handle single-char tokens:
            if formula[index] == ' ' or formula[index] == '\t' or formula[index] == '\n':
                # Skip whitespace.
                index += 1
            elif formula[index] == '(':
                # Open parenthesis... process recursively and continue.
                (inner_tokens, updated_index) = _tokenize_(formula, index + 1)
                assert updated_index > index
                index = updated_index
                tokens.append(inner_tokens)
            elif formula[index] == ')':
                # Closing parenthesis... if this is the "top" execution, this is
                # an error. Otherwise it means we are done parsing this "level".
                if top:
                    raise Exception(f"Unexpected `)` at position {index} in `{formula}`.")
                else:
                    return (tokens, index + 1)
            elif formula[index] == ',':
                # Commas can separate function arguments, but are just copied.
                tokens.append(str(formula[index]))
                index += 1
            elif formula[index] == '*' or formula[index] == '+' or formula[index] == '-' or formula[index] == '/':
                # Arithmetic operators are also just tokens.
                tokens.append(str(formula[index]))
                index += 1
            else:
                number_match = re.search("^\d+", formula[index:])
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
            return (tokens, index)
        else:
            # However, in any other place this is an error.
            raise Exception(f"Missing `)` in formula `{formula}`.")
            
            

    (tokens, index) = _tokenize_(formula, 0, top=True)
    assert index == len(formula), \
        f"Unexpected trailing tokens: {formula[index:]}"
    return tokens
        
def parse_bma_tokens(tokens, ids):
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

    def _last_token_at_(tokens, find):
        """
            Find the last occurence of any token in the `find` list.
            If no such token is found, returns `None`.
        """        
        for i in range(len(tokens)):            
            if tokens[-i-1] in find:
                return len(tokens) - i - 1
        return None

    def _split_by_token_(tokens, delimiter):
        result = []
        sublist = []
        for token in tokens:
            if token == delimiter:
                result.append(sublist)
                sublist = []
            else:
                sublist.append(token)
        result.append(sublist)
        return result

    def _parse_literal_(tokens, ids):
        # There are four options for a literal:
        if len(tokens) == 0:
            raise Exception(f"Unexpected empty literal.")
        if len(tokens) == 1 and type(tokens[0]) == list:
            # This is a "redundant" parenthesis block, which is just transparently
            # forwarded to a new parser call.
            return _parse_tokens_(tokens[0], ids)
        if len(tokens) == 1 and type(tokens[0]) == int:
            # Integer literals are just copied.
            return tokens[0]
        if len(tokens) == 2 and tokens[0] == "var" and type(tokens[1]) == list:
            # Variable literals must be followed by exactly one subtree
            # with one integer ID inside. Anything else is an error.
            if len(tokens[1]) == 1 and type(tokens[1][0]) == int:
                if tokens[1][0] not in ids:
                    raise Exception(f"Unknown variable id `{tokens[1][0]}`.")
                return ids[tokens[1][0]]
            else:
                raise Exception(f"Malformed tokens for variable literal: {tokens}.")
        if len(tokens) == 2 and type(tokens[0]) == str and type(tokens[1]) == list:
            # Function calls are similar, but they don't start with "var", but rather
            # an arbitrary name.
            f_name = tokens[0]
            # Split the arguments by comma.
            f_args = _split_by_token_(tokens[1], ",")
            # And parse each argument separately.
            f_args = [_parse_tokens_(x, ids) for x in f_args]
            return {
                'fn': f_name,
                'args': f_args,
            }

        # If we got here, it means the tokens did not pass any criteria for any literal type.
        raise Exception(f"Malformed literal: {tokens}.")

    def _parse_multiplication_(tokens, ids):
        op_index = _last_token_at_(tokens, ['*', '/'])
        if op_index == None:
            # There is no multiplication, we can continue straight to addition.
            return _parse_literal_(tokens, ids)
        else:
            # Split based on op_index and recursively continue in both halves.
            return {
                'op': tokens[op_index],
                'left': _parse_multiplication_(tokens[:op_index], ids),
                'right': _parse_literal_(tokens[op_index+1:], ids)
            }
    
    def _parse_addition_(tokens, ids):
        op_index = _last_token_at_(tokens, ['+', '-'])
        if op_index == None:
            # There is not addition, continue straits to literals.
            return _parse_multiplication_(tokens, ids)
        else:
            # Split based on op_index and recursively continue.
            return {
                'op': tokens[op_index],
                'left': _parse_addition_(tokens[:op_index], ids),
                'right': _parse_multiplication_(tokens[op_index+1:], ids)
            }

    def _parse_tokens_(tokens, ids):
        return _parse_addition_(tokens, ids)
    
    
    return _parse_tokens_(tokens, ids)

def bma_formula_support_set(formula):
    """
        Return the names of variables which appear in this parsed BMA formula.
    """
    if type(formula) == int:
        return set()
    if type(formula) == str:
        return set([formula])
    if 'op' in formula:
        support = bma_formula_support_set(formula['left'])
        return support.union(bma_formula_support_set(formula['right']))
    if 'fn' in formula:
        support = set()
        for arg in formula['args']:
            support = support.union(bma_formula_support_set(arg))
        return support
    raise Exception(f"Invalid formula: {formula}")

def bma_formula_eval(formula, valuation):
    """
        Evaluate the given formula using the values in the provided valuation dictionary.

        The valuation should contain both the "nominal level" of the input, as well as
        target-normalized `Fraction` value that will actually be used.

        The result is a floating point number that you should convert to an integer
        and clip based on the allowed variable range.
    """
    if type(formula) == int:
        return Fraction(numerator=formula, denominator=1)
    if type(formula) == str:
        # Use the normalized fraction level from the valuation, not the nominal level.
        return valuation[formula][1]
    if 'op' in formula:
        left = bma_formula_eval(formula['left'], valuation)
        right = bma_formula_eval(formula['right'], valuation)
        if formula['op'] == '+':
            return left + right
        if formula['op'] == '-':
            return left - right
        if formula['op'] == '*':
            return left * right
        if formula['op'] == '/':
            return left / right
    if 'fn' in formula:
        args = [bma_formula_eval(arg, valuation) for arg in formula['args']]
        if formula['fn'] == "min":
            return min(args)
        if formula['fn'] == "max":
            return max(args)
        if formula['fn'] == "avg":
            return statistics.mean(args)
        if formula['fn'] == "ceil":
            assert len(args) == 1, \
                f"Ceil function expects one argument, but {len(args)} were found."
            return math.ceil(args[0])
        if formula['fn'] == "floor":
            assert len(args) == 1, \
                f"Floor function expects one argument, but {len(args)} were found."
            return math.floor(args[0])
        raise Exception(f"Unknown function: {formula['fn']}.")

    raise Exception(f"Invalid formula: {formula}")

def build_function_table(var, formula, ids, levels):
    def _expand_support_(target_variable, valuation, remaining_support, levels):
        if len(remaining_support) == 0:
            return [valuation.copy()]
        
        variable = remaining_support[0]
        remaining_support = remaining_support[1:]
        results = []
        for level in range(levels[variable] + 1):
            # The second value is the "normalized level" the authors claim they are using
            # to better align the domains of the two variables.
            # The normalization is slightly simplified since we require all variables to start at 0,
            # hence we don't have to account for different low bounds, only different max bounds.
            valuation[variable] = (level, Fraction(numerator=(level * levels[target_variable]), denominator=levels[variable]))            
            results = results + _expand_support_(target_variable, valuation, remaining_support, levels)
            del valuation[variable]
        return results

    support = bma_formula_support_set(formula)
    inputs = _expand_support_(var, {}, list(support), levels)
    table = []
    for i in inputs:
        result_fraction = bma_formula_eval(formula, i)
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
            result_decimal = Decimal(result_fraction.numerator)/Decimal(result_fraction.denominator)        
            result = int(result_decimal.quantize(Decimal(0), rounding=ROUND_HALF_UP))        
        result = max(0, result)
        result = min(result, levels[var])
        table.append((i, result))
    return table

def bma_file_to_petri_net(path, unitary=False):
    return bma_string_to_petri_net(Path(path).read_text(), unitary)

def bma_string_to_petri_net(model_string, unitary=False):
    return bma_model_to_petri_net(json.loads(model_string), unitary)

def canonical_name(v_name, v_id):
    """
        Compute a canonical name which will be unique within the BMA model and not contain any
        special characters (as opposed to BMA, which allows both special characters and duplicate
        names).
    """
    v_name = re.sub("[^0-9a-zA-Z_]", "_", v_name)
    return f"{v_name}_id{str(v_id)}"

def bma_model_to_petri_net(model_json, unitary=False):
    model = model_json['Model']

    # Read max. levels for every variable.
    levels = {}
    ids = {}
    regulators = {}
    targets = {}
    zero_constants = []
    for variable in model['Variables']:
        v_name = variable['Name']
        v_id = variable['Id']
        v_name = canonical_name(v_name, v_id)
        assert v_name not in levels.keys(), \
            f"Duplicate name after normalization: {v_name}."
        # For some reason, some BMA models use strings here instead of numbers?!
        v_min = int(variable['RangeFrom'])
        v_max = int(variable['RangeTo'])
        assert v_min == 0, \
            f"Only variables with min. level 0 are supported. `{v_name}` has min. level {v_min}."                
        levels[v_name] = v_max
        regulators[v_name] = []
        targets[v_name] = []
        ids[int(v_id)] = v_name
        if v_max == 0:
            # Override zero constants to normal variables.
            # Later as we are building the update functions, we will ensure
            # that the value always tends towards zero.
            zero_constants.append(v_name)
            levels[v_name] = 1

    for reg in model['Relationships']:
        source = ids[reg['FromVariable']]
        target = ids[reg['ToVariable']]
        sign = '?'
        if reg['Type'] == 'Activator':
            sign = '+'
        if reg['Type'] == 'Inhibitor':
            sign = '-'
        assert sign != '?', \
            f"Unknown type of regulation: {reg}."
        regulators[target].append((source, sign))
        targets[source].append((target, sign))

    encoding = SymbolicEncoding(levels)
    implicants = {}

    for variable in model['Variables']:
        v_name = variable['Name']
        v_id = variable['Id']
        v_name = canonical_name(v_name, v_id)
        v_formula = variable['Formula']        
        if v_name in zero_constants:
            # Zero constants don't have a formula but can be only zero.
            formula = 0
        elif len(v_formula) == 0:
            positive = []
            negative = []
            for (regulator, sign) in regulators[v_name]:
                if sign == '+':
                    positive.append(regulator)
                elif sign == '-':
                    negative.append(regulator)
            if len(positive) == 0 and len(negative) == 0:
                # This is an undetermined input, in which case we set it to zero, because that's what BMA does.
                formula = 0
            else:
                if len(negative) == 0:
                    n_avr = 0
                else:
                    n_avr = { 'fn': 'avg', 'args': negative }
                if len(positive) == 0:
                    # This does not make much sense, because it means any variable with only negative
                    # regulators is ALWAYS a constant zero. But this is how BMA seems to be doint it, so
                    # that's what we are doing as well...
                    p_avr = 0
                else:
                    p_avr = { 'fn': 'avg', 'args': positive }

                formula = { 'op': '-', 'left': p_avr, 'right': n_avr }
            #print(f"Implicit formula for {v_name}: {len(positive)} vs. {len(negative)} => {formula}")
        else:
            tokens = tokenize_bma_formula(v_formula)
            formula = parse_bma_tokens(tokens, ids)

        table = build_function_table(v_name, formula, ids, levels)        
        implicants[v_name] = encoding.build_implicant_bdds(v_name, table)
    
    # Now we can start building the Petri net.
    net = nx.DiGraph()

    # Create places for all variables.
    place_count = 0
    for var in levels:
            for i in range(0, levels[var]+1):
                net.add_node(f"p{var}_b{i}", kind="place")
                place_count += 1

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

    # print(f"Created {tr_id} transitions and {place_count} places for {len(implicants)} model variables.")

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

if __name__ == '__main__':
    print(bma_file_to_petri_net(sys.argv[1], unitary=True))
    
