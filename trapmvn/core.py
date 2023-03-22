from __future__ import annotations

from trapmvn.representation.petri_net import Petri_Net
from trapmvn.representation.symbolic import Symbolic_Model, Symbolic_Function
from trapmvn.representation.bma import BMA_Model
from trapmvn.representation.sbml import SBML_Model
from typing import Dict, List, Tuple, Union, Callable, Optional
from clingo import Control, Model, SolveHandle, Symbol, String

class Trap:
    mirror: bool
    model: Model
    net: Petri_Net

    def __init__(self, net: Petri_Net, model: Model, mirror: bool):
        self.model = model
        self.mirror = mirror
        self.net = net

    def decode(self) -> Dict[str, List[int]]:
        if self.mirror:
            return _decode_result_mirror(self.model, self.net)
        else:
            return _decode_result(self.model, self.net)

    def contains(self, variable: str, value: int) -> bool:
        result = self.model.contains(String(self.net.place_map[variable][value]))
        if self.mirror:
            result = not result
        return result

    def __str__(self) -> str:
        return str(self.decode())

def trapmvn(
    model: Union[Petri_Net, Symbolic_Model, SBML_Model, BMA_Model],
    limit: Optional[int] = None,
    semantics: str = "unitary",
    problem: str = "min",
    fixed_point_method: str = "deadlock"
) -> List[Trap]:
    """
        Analyse the given `model`, returning a list of trap spaces.

        Other parameters:
         - `limit`; Max. number of spaces that should be enumerated (can be `None`).
         - `semantics in ["unitary", "general"]`; The type of used value update.
         - `problem in ["min", "max", "fix"]`; Determine whether to compute minimal/maximal
            trap spaces, or fixed-points.
         - `fixed_point_method in ["deadlock", "siphon"]`; Determine what algorithm should
            be used for fixed-point computation.
    """
    results = []
    def save_result(x):
        results.append(x)
        return limit == None or len(results) < limit        
    trapmvn_async(model, save_result, semantics, problem, fixed_point_method)
    return results

def trapmvn_async(
    model: Union[Petri_Net, Symbolic_Model, SBML_Model, BMA_Model],
    on_solution: Callable[[Trap], bool],
    semantics: str = "unitary",
    problem: str = "min",
    fixed_point_method: str = "deadlock"
):
    """
        Asynchronously analyse the given model, returning results through the `on_solution` 
        callback. See `trapmvn` for explanation of other parameters.
    """
    petri_net = _normalize_model(model, semantics)
    ctl = _encode_asp_problem(petri_net, problem, fixed_point_method)
    ctl.ground()
    result = ctl.solve(yield_=True)
    if type(result) == SolveHandle:
        with result as iterator:
            for clingo_model in iterator:                    
                if problem == "fix" and fixed_point_method == "deadlock":
                    space = Trap(petri_net, clingo_model, mirror=False)
                else:
                    space = Trap(petri_net, clingo_model, mirror=True)
                if not on_solution(space):
                    break
    # Else: unsat, hence we don't do anything.

def _normalize_model(model: Union[Petri_Net, Symbolic_Model, SBML_Model, BMA_Model], semantics: str) -> Petri_Net:
    """
        Convert arbitrary model type to a valid petri net.
    """
    assert semantics in ["unitary", "general"]    
    if type(model) == SBML_Model or type(model) == BMA_Model:
        return _optimized_translation(model, semantics == "unitary")
    if type(model) == Petri_Net:
        return model
    if type(model) == Symbolic_Model:
        symbolic_model = model
        return Petri_Net.build(symbolic_model, semantics == "unitary")
    else:
        raise TypeError("Unknown type of model.")

def _optimized_translation(model: Union[SBML_Model, BMA_Model], unitary: bool) -> Petri_Net:
    """
        Test multiple encodings for each function and then pick the one which produces the least amount of implicants.
    """
    nets = []
    for variable in model.variables:
        candidate_encodings = [Symbolic_Function.from_model(model, variable)]
        arity = len(model.functions[variable].inputs) if variable in model.functions else 0
        for i in range(1, arity):
            candidate_encodings.append(Symbolic_Function.from_model(model, variable, seed=i))
        candidate_networks = [Petri_Net.for_function(f, unitary) for f in candidate_encodings]        
        candidate_networks = sorted(candidate_networks, key=lambda pn: pn.count_implicants(variable))
        nets.append(candidate_networks[0])
        

    pn = nets[0]
    for extra in nets[1:]:
        pn = pn.merge(extra)
    return pn


def _decode_result(result: Model, petri_net: Petri_Net) -> Dict[str, List[int]]:
    """
        Decode a "positive" result of clingo computation. That is, the trap space
        is represented through positive atoms.
    """
    space: Dict[str, List[int]] = { var: [] for var in petri_net.place_map.keys() }
    for atom in result.symbols(atoms=True):
        atom_str = str(atom)
        (variable, level) = petri_net.places[atom_str]        
        space[variable].append(level)
    # Sort levels to ensure canonical representation.
    space = { var:sorted(values) for var, values in space.items() }
    return space

def _decode_result_mirror(result: Model, petri_net: Petri_Net) -> Dict[str, List[int]]:
    """
        Decore a "negative" result of clingo computation. That is, the trap space
        is represented through the inverse of positive atoms.
    """
    space: Dict[str, List[int]] = { var: list(range(len(places))) for var, places in petri_net.place_map.items() }
    for atom in result.symbols(atoms=True):
        atom_str = str(atom)
        (variable, level) = petri_net.places[atom_str]
        space[variable].remove(level)        
    return space

def _encode_asp_problem(
    petri_net: Petri_Net,
    problem: str = "min",
    fixed_point_method: str = "deadlock"
) -> Control:
    assert problem in ["min", "max", "fix"], f"Unknown problem type: {problem}."
    assert fixed_point_method in ["siphon", "deadlock"], f"Unknown fixed-point algorithm: {fixed_point_method}."

    var_set = set([ variable for variable, _level in petri_net.places.values() ])
    variables = sorted(var_set) # Sort to ensure determinism.

    dom_mod = "--dom-mod=5" if problem == "max" else "--dom-mod=3"    
    ctl = Control(["0", "--heuristic=Domain", "--enum-mod=domRec", dom_mod])

    if problem == "fix" and fixed_point_method == "deadlock":
        for var in variables:
            # Declare each place.
            for place in petri_net.place_map[var]:
                ctl.add(f"{{{place}}}.")
            
            # Add deadlock condition.
            ctl.add(f"1 {{{'; '.join(petri_net.place_map[var])}}} 1.")
        
        for var in variables:
            # Build a rule for each PN transition.
            for (s_level, _t_level), implicant_list in petri_net.implicants[var].items():
                s_place = petri_net.place_map[var][s_level]
                for implicants in implicant_list:
                    rule = [s_place] + implicants
                    ctl.add(f":- {'; '.join(rule)}.")        
    else:
        for var in variables:
            places = petri_net.place_map[var]
            
            # Declare each place.
            for place in places:
                ctl.add(f"{{{place}}}.")
                
            # Add conflict-free condition.
            ctl.add(f":- {'; '.join(places)}.")
            
            if problem == "fix":
                # Ensure only one atom can be part of the solution.
                ctl.add(f"{{{'; '.join(places)}}} >= {len(places) - 1}.")
    
        if problem == "max":
            # Add maximality condition.
            ctl.add(f"{'; '.join(petri_net.places.keys())}.")

        for var in variables:
            # Build a rule for each PN transition.
            for (s_level, t_level), implicant_list in petri_net.implicants[var].items():
                s_place = petri_net.place_map[var][s_level]
                t_place = petri_net.place_map[var][t_level]
                for implicants in implicant_list:                    
                    rule_left = '; '.join([s_place] + implicants)
                    ctl.add(f"{rule_left} :- {t_place}.")
                    
    return ctl