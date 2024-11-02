from __future__ import annotations
import random

from trapmvn.representation.sbml import (
    SBML_Model,
    SBML_Proposition,
    SBML_Term,
    SBML_Function,
    SBML_Expression,
    CmpOp,
)

def randomly_expand_boolean_variables(model: SBML_Model, seed: int = 0):
    for (var, max_level) in model.variables.copy().items():
        if max_level == 1:
            randomly_expand_variable_domain(model, var, seed)

def randomly_expand_variable_domain(model: SBML_Model, variable: str, seed: int = 0):
    assert variable in model.variables, "Variable not found in model."
    assert model.variables[variable] == 1, "Only Boolean variables can be expanded."

    # Find all targets regulated by the given variable.
    targets = set()
    for func_var, func in model.functions.items():
        if variable in func.inputs:
            targets.add(func_var)

    rng = random.Random(seed)
    target_list = sorted(targets)
    rng.shuffle(target_list)
    new_max_level = len(targets)

    if new_max_level < 2:
        # Domain cannot be expanded further.
        return

    # Update domain size.
    model.variables[variable] = new_max_level

    # Update propositions that use variable in update functions to make them multi-valued.
    for i, target_var in enumerate(target_list):
        threshold = i + 1  # Threshold is the target variable index, plus one.
        model.functions[target_var] = rebuild_function_with_threshold(
            model.functions[target_var], variable, threshold
        )

    # For the variable whose domain is affected, we just set all non-zero terms
    # to the max value. This only "works" in the unitary update, but the whole generator
    # is quite biologically-inspired anyway.

    func = model.functions[variable]
    if func.default_result == 1:
        func.default_result = new_max_level
    else:
        assert func.default_result == 0

    for term in func.terms:
        if term.result == 1:
            term.result = new_max_level
        else:
            assert term.result == 0

    # Done


def rebuild_function_with_threshold(
    func: SBML_Function, var: str, threshold: int
) -> SBML_Function:
    return SBML_Function(
        func.inputs.copy(),
        func.output,
        func.default_result,
        [rebuild_term_with_threshold(t, var, threshold) for t in func.terms],
    )


def rebuild_term_with_threshold(
    term: SBML_Term, var: str, threshold: int
) -> SBML_Expression:
    if type(term.expression) == SBML_Expression:
        new_expr = rebuild_expression_with_threshold(term.expression, var, threshold)
    elif type(term.expression) == SBML_Proposition:
        new_expr = rebuild_proposition_with_threshold(term.expression, var, threshold)
    else:
        raise NotImplementedError(f"Unknown expression {term.expression}.")
    return SBML_Term(term.result, new_expr)


def rebuild_expression_with_threshold(
    expr: SBML_Expression, var: str, threshold: int
) -> SBML_Expression:
    new_args = []
    for arg in expr.arguments:
        if type(arg) == SBML_Proposition:
            new_args.append(rebuild_proposition_with_threshold(arg, var, threshold))
        elif type(arg) == SBML_Expression:
            new_args.append(rebuild_expression_with_threshold(arg, var, threshold))
        else:
            raise NotImplementedError(f"Unknown expression {expr}.")
    return SBML_Expression(expr.operator, new_args)


def rebuild_proposition_with_threshold(
    prop: SBML_Proposition, var: str, threshold: int
) -> SBML_Proposition:
    """
    Update this proposition to use the given threshold, assuming it uses the given variable.

    A changed proposition needs to be boolean.
    """
    if prop.variable != var:
        return prop

    # x < 0 => contradiction => unchanged
    if prop.operator == CmpOp.LT and prop.constant == 0:
        return prop
    # x < 1 => x < threshold
    if prop.operator == CmpOp.LT and prop.constant == 1:
        return SBML_Proposition(prop.variable, CmpOp.LT, threshold)
    # x <= 0 => x < 1 => x < threshold
    if prop.operator == CmpOp.LEQ and prop.constant == 0:
        return SBML_Proposition(prop.variable, CmpOp.LT, threshold)
    # x <= 1 => tautology => x <= max_level
    if prop.operator == CmpOp.LEQ and prop.constant == 1:
        return SBML_Proposition(prop.variable, CmpOp.GEQ, 0)
    # x == 0 => x < 1 => x < threshold
    if prop.operator == CmpOp.EQ and prop.constant == 0:
        return SBML_Proposition(prop.variable, CmpOp.LT, threshold)
    # x == 1 => x > 0 => x >= threshold
    if prop.operator == CmpOp.EQ and prop.constant == 1:
        return SBML_Proposition(prop.variable, CmpOp.GEQ, threshold)
    # x >= 0 => tautology => unchanged
    if prop.operator == CmpOp.GEQ and prop.constant == 0:
        return prop
    # x >= 1 => x > 0 => x >= threshold
    if prop.operator == CmpOp.GEQ and prop.constant == 1:
        return SBML_Proposition(prop.variable, CmpOp.GEQ, threshold)
    # x > 0 => x >= threshold
    if prop.operator == CmpOp.GT and prop.constant == 0:
        return SBML_Proposition(prop.variable, CmpOp.GEQ, threshold)
    # x > 1 => contradiction => x > max_level
    if prop.operator == CmpOp.GT and prop.constant == 1:
        return SBML_Proposition(prop.variable, CmpOp.LT, 0)

    raise NotImplemented(f"Invalid proposition: {prop}")
