from __future__ import annotations

import re
import random
from biodivine_aeon import BddVariableSetBuilder, RegulatoryGraph # type: ignore
from trapmvn.representation.petri_net import expand_universal_integers
from trapmvn.representation.sbml import SBML_Proposition, SBML_Expression, SBML_Term, SBML_Function, CmpOp, LogicOp, SBML_Model
from trapmvn.representation.bma import BMA_Model
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, List, Tuple, Union
    from biodivine_aeon import BddVariableSet, Bdd, BddVariable # type: ignore

class Symbolic_Function:
    ctx: BddVariableSet
    variable: str
    booleans: Dict[str, BddVariable]
    integers: Dict[str, List[BddVariable]]
    function: List[Bdd]
    function_inputs: List[str]

    def __init__(
            self,
            ctx: BddVariableSet,
            variable: str,
            booleans: Dict[str, BddVariable],
            integers: Dict[str, List[BddVariable]],
            function: List[Bdd],
            function_inputs: List[str],
    ):
        self.ctx = ctx
        self.variable = variable
        self.booleans = booleans
        self.integers = integers
        self.function = function
        self.function_inputs = function_inputs

    @staticmethod
    def from_model(model: Union[SBML_Model, BMA_Model], variable: str, seed: int | None = None) -> Symbolic_Function:
        if type(model) == SBML_Model:
            return Symbolic_Function.from_sbml(model, variable, seed)
        if type(model) == BMA_Model:
            return Symbolic_Function.from_bma(model, variable, seed)
        raise TypeError("Unknown model type")

    @staticmethod
    def from_sbml(model: SBML_Model, variable: str, seed: int | None = None) -> Symbolic_Function:
        if variable not in model.functions:
            # Inputs don't have regulators
            regulators = []
        else:
            regulators = model.functions[variable].inputs.copy()

        relevant_variables = { input: model.variables[input] for input in regulators }
        relevant_variables[variable] = model.variables[variable]

        (bdd_vars, booleans, integers) = build_symbolic_context(relevant_variables, seed)
        
        implicant_bdds = [bdd_vars.mk_const(False) for _ in range(relevant_variables[variable] + 1)]
        if variable not in model.functions:
            # This variable is a constant. This means change of value is never possible.
            return Symbolic_Function(bdd_vars, variable, booleans, integers, implicant_bdds, regulators)

        sbml_function = model.functions[variable]            
        default_term = bdd_vars.mk_const(True)

        for term in sbml_function.terms:
            # Build term BDD
            term_bdd = term.evaluate_symbolic(bdd_vars, booleans, integers)
            # Clean up any invalid encoding values.
            for i in sbml_function.inputs:
                if i in integers:
                    term_bdd = clean_encoding(bdd_vars, term_bdd, integers[i])
            # Add the BDD to existing conditions for the result level.
            implicant_bdds[term.result] = implicant_bdds[term.result].l_or(term_bdd)
            # And remove it from conditions for the default term.
            default_term = default_term.l_and_not(term_bdd)

        # Finally, also clean up the default term.
        for i in sbml_function.inputs:
            if i in integers:
                default_term = clean_encoding(bdd_vars, default_term, integers[i])

        # And add it to the correct level BDD.
        implicant_bdds[sbml_function.default_result] = implicant_bdds[sbml_function.default_result].l_or(default_term)

        return Symbolic_Function(bdd_vars, variable, booleans, integers, implicant_bdds, regulators)

    @staticmethod
    def from_bma(model: BMA_Model, variable, seed: int | None = None) -> Symbolic_Function:        
        regulators = model.functions[variable].inputs.copy()
        relevant_variables = { input: model.variables[input] for input in regulators }
        relevant_variables[variable] = model.variables[variable]

        (bdd_vars, booleans, integers) = build_symbolic_context(relevant_variables, seed)
        
        function_table = model.build_function_table(variable)

        implicant_bdds = [bdd_vars.mk_const(False) for _ in range(relevant_variables[variable] + 1)]
        for (valuation, output) in function_table:
            clause = {}
            for (var, level) in valuation.items():                    
                if var in booleans:
                    clause[booleans[var]] = level == 1
                else:
                    for i in range(relevant_variables[var] + 1):
                        level_var = integers[var][i]
                        clause[level_var] = (i == level)
            clause_bdd = bdd_vars.mk_conjunctive_clause(clause)
            implicant_bdds[output] = implicant_bdds[output].l_or(clause_bdd)

        return Symbolic_Function(bdd_vars, variable, booleans, integers, implicant_bdds, regulators)

class Symbolic_Model:
    ctx: BddVariableSet
    booleans: Dict[str, BddVariable]
    integers: Dict[str, List[BddVariable]]
    levels: Dict[str, int]
    functions: Dict[str, List[Bdd]]
    function_inputs: Dict[str, List[str]]

    def __init__(
        self, 
        ctx: BddVariableSet, 
        booleans: Dict[str, BddVariable], 
        integers: Dict[str, List[BddVariable]],
        levels: Dict[str, int],
        functions: Dict[str, List[Bdd]],
        function_inputs: Dict[str, List[str]],
    ):
        self.ctx = ctx
        self.booleans = booleans
        self.integers = integers
        self.levels = levels
        self.functions = functions
        self.function_inputs = function_inputs

    @staticmethod
    def from_sbml(model: SBML_Model) -> Symbolic_Model:
        (bdd_vars, booleans, integers) = build_symbolic_context(model.variables)
        
        levels = model.variables.copy()
        functions = {}        
        for variable in model.variables:
            implicant_bdds = [bdd_vars.mk_const(False) for _ in range(levels[variable] + 1)]
            if variable not in model.functions:
                # This variable is a constant. This means change of value is never possible.
                functions[variable] = implicant_bdds
                continue

            sbml_function = model.functions[variable]            
            default_term = bdd_vars.mk_const(True)

            for term in sbml_function.terms:
                # Build term BDD
                term_bdd = term.evaluate_symbolic(bdd_vars, booleans, integers)
                # Clean up any invalid encoding values.
                for i in sbml_function.inputs:
                    if i in integers:
                        term_bdd = clean_encoding(bdd_vars, term_bdd, integers[i])
                # Add the BDD to existing conditions for the result level.
                implicant_bdds[term.result] = implicant_bdds[term.result].l_or(term_bdd)
                # And remove it from conditions for the default term.
                default_term = default_term.l_and_not(term_bdd)

            # Finally, also clean up the default term.
            for i in sbml_function.inputs:
                if i in integers:
                    default_term = clean_encoding(bdd_vars, default_term, integers[i])

            # And add it to the correct level BDD.
            implicant_bdds[sbml_function.default_result] = implicant_bdds[sbml_function.default_result].l_or(default_term)
            functions[sbml_function.output] = implicant_bdds

        function_inputs = { v:f.inputs for (v,f) in model.functions.items() }

        return Symbolic_Model(bdd_vars, booleans, integers, levels, functions, function_inputs)
            

        
    @staticmethod
    def from_bma(model: BMA_Model) -> Symbolic_Model:        
        (bdd_vars, booleans, integers) = build_symbolic_context(model.variables)
        
        levels = model.variables.copy()
        functions = {}
        for variable in model.variables:
            function_table = model.build_function_table(variable)

            implicant_bdds = [bdd_vars.mk_const(False) for _ in range(levels[variable] + 1)]
            for (valuation, output) in function_table:
                clause = {}
                for (var, level) in valuation.items():                    
                    if var in booleans:
                        clause[booleans[var]] = level == 1
                    else:
                        for i in range(levels[var] + 1):
                            level_var = integers[var][i]
                            clause[level_var] = (i == level)
                clause_bdd = bdd_vars.mk_conjunctive_clause(clause)
                implicant_bdds[output] = implicant_bdds[output].l_or(clause_bdd)
            
            functions[variable] = implicant_bdds

        function_inputs = { v:f.inputs for (v,f) in model.functions.items() }

        return Symbolic_Model(bdd_vars, booleans, integers, levels, functions, function_inputs)

    def to_sbml(self) -> SBML_Model:
        variables = self.levels.copy()
        functions = {}

        boolean_bdd_variables = set(self.booleans.values())

        for var in self.functions:
            bdds = self.functions[var]
            # Add the result level to each BDD.
            # Ideally, we'd like to have the largest BDD as the "default" level,
            # but most tools can't deal with non-zero default level, so we just keep 
            # it at that :(.
            level_bdds: List[Tuple[int, Bdd]] = [ (i, bdds[i]) for i in range(len(bdds))]
            level_bdds = list(reversed(level_bdds))
            (default_level, bdd) = level_bdds.pop()
            assert default_level == 0
                
            if all([bdd.is_false() for bdd in bdds]):
                # This is an unknown constant - all BDDs are false.
                # For constants, we don't create the function at all.
                continue

            if any([bdd.is_true() for bdd in bdds]):
                # This is a fi29xed constant. Just output default term and nothing else.
                for i in range(len(bdds)):
                    if bdds[i].is_true():
                        default_level = i
                function = SBML_Function(
                    inputs=[],
                    output=var,
                    default_result=default_level,
                    terms=[]
                )
                functions[var] = function
                continue

            inputs = set()
            terms = []
            for (result_level, bdd) in level_bdds:
                if bdd.is_false():
                    continue
                disjunction: List[Union[SBML_Expression, SBML_Proposition]] = []
                for bdd in expand_universal_integers(self.ctx, self.integers, self.function_inputs[var], bdd):                    
                    for clause in bdd.list_sat_clauses():
                        literals: List[Union[SBML_Expression, SBML_Proposition]] = []
                        for (bdd_var, value) in clause:
                            if bdd_var in boolean_bdd_variables:
                                # For Boolean variables, the name should be the same,
                                # just string the first 'p'.
                                name = self.ctx.name_of(bdd_var)[1:]
                                inputs.add(name)
                                literals.append(SBML_Proposition(name, CmpOp.EQ, int(value)))
                            else:                                
                                # For integer variables, we have to deconstruct the name.
                                # We also only take positive literals, as negative literals
                                # are there mostly to ensure correct encoding.
                                if value:
                                    m = re.match('p(.+?)_b(\\d+)', self.ctx.name_of(bdd_var))
                                    assert m is not None
                                    name = m[1]
                                    value = int(m[2])
                                    inputs.add(name)
                                    literals.append(SBML_Proposition(name, CmpOp.EQ, value))                                    
                        disjunction.append(SBML_Expression(LogicOp.AND, literals))   
                if len(disjunction) > 0:
                    term = SBML_Term(result_level, SBML_Expression(LogicOp.OR, disjunction))
                    terms.append(term)
                
            function = SBML_Function(
                    inputs=sorted(list(inputs)),
                    output=var,
                    default_result=default_level,
                    terms=terms
                )
            functions[var] = function
        
        return SBML_Model(variables, functions)

    def infer_influence_graph(self) -> RegulatoryGraph:
        """
        Compute an influence graph (represented using AEON's `RegulatoryGraph`) which
        contains the exact monotonicity properties of each update function.
        """
        variables = list(self.levels.keys())
        rg = RegulatoryGraph(variables)

        for var in variables: 
            var_function = self.functions[var]
            # Infer "observable" regulators for `var`.
            supports = set()
            for bdd in var_function:
                supports.update(bdd.support_set())
            regulators = set()
            for reg, v in self.booleans.items():
                if v in supports:
                    regulators.add(reg)
            for reg, l in self.integers.items():
                for v in l:
                    if v in supports:
                        regulators.add(reg)
            # Infer monotonicity of each regulator.
            for reg in regulators:
                # To test activation, we literally verify
                # that given `f(x) = y` where `x[reg] = a`,
                # for every `z > y` and `b > a`, we don't
                # have `f(x[reg=b])=z`.
                is_activation = True
                if reg in self.booleans:
                    # If the regulator is a Boolean, there is only
                    # one symbolic variable to test.
                    reg_var = self.booleans[reg]
                    for target_low_level in range(self.levels[var] + 1):
                        # Regulator is high in the lower target value.
                        low_level_bdd = var_function[target_low_level].var_restrict(reg_var, True)
                        for target_higher_level in range(target_low_level + 1, self.levels[var] + 1):
                            # Regulator is low in the higher target value.
                            high_level_bdd = var_function[target_higher_level].var_restrict(reg_var, False)
                            if not low_level_bdd.l_and(high_level_bdd).is_false():
                                # There two things cannot happen for the same combination of
                                # remaining inputs.
                                is_activation = False
                else:
                    reg_vars = self.integers[reg]
                    for reg_high in range(self.levels[reg] + 1):
                        # reg_high is the level from which the regulator
                        # should decrease when ruling out activation.
                        for target_low_level in range(self.levels[var] + 1):
                            # Regulator is at `reg_high` level in the lower target value.
                            low_level_bdd = var_function[target_low_level]
                            low_level_bdd = low_level_bdd.var_select(reg_vars[reg_high], True)
                            low_level_bdd = low_level_bdd.project(reg_vars)
                            for target_higher_level in range(target_low_level + 1, self.levels[var] + 1):
                                # Regulator is at any lower level in the higher target value.
                                high_level_bdd = var_function[target_higher_level]
                                reg_is_lower = self.ctx.mk_disjunctive_clause({ x:True for x in reg_vars[:reg_high] })
                                high_level_bdd = high_level_bdd.l_and(reg_is_lower).project(reg_vars)                                
                                if not low_level_bdd.l_and(high_level_bdd).is_false():                                    
                                    is_activation = False                                    

                # Inhibition is the same, except the regulator must not go "up"
                # in the higher levels instead of "down"
                is_inhibition = True
                if reg in self.booleans:
                    reg_var = self.booleans[reg]
                    for target_low_level in range(self.levels[var] + 1):
                        # Regulator is low at the lower target level.
                        low_level_bdd = var_function[target_low_level].var_restrict(reg_var, False)
                        for target_higher_level in range(target_low_level + 1, self.levels[var] + 1):
                            # Regulator is high at the higher target level.
                            high_level_bdd = var_function[target_higher_level].var_restrict(reg_var, True)
                            if not low_level_bdd.l_and(high_level_bdd).is_false():
                                is_inhibition = False
                else:
                    reg_vars = self.integers[reg]
                    for reg_low in range(self.levels[reg] + 1):
                        for target_low_level in range(self.levels[var] + 1):
                            low_level_bdd = var_function[target_low_level]
                            low_level_bdd = low_level_bdd.var_select(reg_vars[reg_low], True)
                            low_level_bdd = low_level_bdd.project(reg_vars)
                            for target_higher_level in range(target_low_level + 1, self.levels[var] + 1):
                                high_level_bdd = var_function[target_higher_level]
                                reg_is_higher = self.ctx.mk_disjunctive_clause({ x:True for x in reg_vars[reg_low+1:] })
                                high_level_bdd = high_level_bdd.l_and(reg_is_higher).project(reg_vars)
                                if not low_level_bdd.l_and(high_level_bdd).is_false():                                    
                                    is_inhibition = False  

                assert not (is_activation and is_inhibition)

                if is_activation:
                    rg.add_regulation({
                            'source': reg,
                            'target': var,
                            'monotonicity': 'activation',
                            'observable': True
                    })
                elif is_inhibition:
                    rg.add_regulation({
                            'source': reg,
                            'target': var,
                            'monotonicity': 'inhibition',
                            'observable': True
                    })
                else:
                    rg.add_regulation({
                            'source': reg,
                            'target': var,
                            'observable': True
                    })
        return rg
                
                


            




def build_symbolic_context(levels: Dict[str, int], seed: int | None = None) -> Tuple[BddVariableSet, Dict[str, BddVariable], Dict[str, List[BddVariable]]]:
    """
        Build the basis of the symbolic encoding (the BDD variables) for the given variable space.
    """
    # Ensures deterministic variable order with optional shuffle.
    if seed is None:
        variables = list(sorted(levels.keys()))
    else:
        rng = random.Random(seed)
        variables = list(levels.keys())
        rng.shuffle(variables)

    # Dictionary which maps Boolean BMA variables to their BDD counterparts.
    booleans = {}
    # Dictionary which maps integer BMA variables to a list of BDD counterparts.
    integers = {}

    bdd_vars_builder = BddVariableSetBuilder()
    for var in variables:
        max_level = levels[var]
        if max_level == 1:
            # Boolean variables are expanded later when implicants are constructed 
            # (it keeps the BDDs smaller).
            v = bdd_vars_builder.make_variable(f"p{var}")
            booleans[var] = v
        else:
            # Integer variables are expanded into `k` distinct Boolean variables immediately.
            integers[var] = bdd_vars_builder.make([f"p{var}_b{x}" for x in range(max_level + 1)])
    return (bdd_vars_builder.build(), booleans, integers)

def clean_encoding(ctx: BddVariableSet, bdd: Bdd, symbolic: List[BddVariable]) -> Bdd:
    """
        Ensure that the BDD only contains properly encoded values of 
        the provided `symbolic` variable encoding.
    """        
    max_level = len(symbolic) - 1

    # Ensure that at most one encoding variable can be true.
    for l in range(0, max_level + 1):
        # l=1 => (a=0 & b=0 & ...)            
        l_is_true = ctx.mk_literal(symbolic[l], True)
        rest = { x:False for x in symbolic if x != symbolic[l] }
        rest_is_false = ctx.mk_conjunctive_clause(rest)
        condition = l_is_true.l_imp(rest_is_false)
        bdd = bdd.l_and(condition)
    # Ensure that at least one encoded variable is true.
    # a=1 | b=1 | c=1 | ...
    at_least_one = { x:True for x in symbolic }
    at_least_one = ctx.mk_disjunctive_clause(at_least_one) 
    bdd = bdd.l_and(at_least_one)
    return bdd
