from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, Set, List, Tuple, Union, Optional
    from trapmvn.representation.symbolic import Symbolic_Model, Symbolic_Function
    from biodivine_aeon import BddVariableSet, Bdd, BddVariable # type: ignore

class Petri_Net:
    """
        A "Petri net" representing encoding the dynamics of a multi-valued network. Note that
        this is not a truly general Petri net. Rather, we just keep a list of implicants
        for each variable level change, as this makes it easier to filter/restrict the Petri net
        after it has been created.
    """
    # Maps network places to their original variable names and levels.
    places: Dict[str, Tuple[str, int]]    
    # Maps network variables to a list of places (indexed by variable level).
    place_map: Dict[str, List[str]]
    # Maps network variables to value change implicants. The key is a value change (e.g. 0 -> 1),
    # the value is a list of implicants (implicant being a list of place names)
    implicants: Dict[str, Dict[Tuple[int, int], List[List[str]]]]

    def __init__(
        self, 
        places: Dict[str, Tuple[str, int]], 
        place_map: Dict[str, List[str]], 
        implicants: Dict[str, Dict[Tuple[int, int], List[List[str]]]]
    ):
        self.places = places
        self.place_map = place_map
        self.implicants = implicants        

    @staticmethod
    def for_function(function: Symbolic_Function, unitary=True) -> Petri_Net:
        places = {}
        place_map: Dict[str, List[str]] = {}
        model_implicants = {}
        boolean_bdd_variables = set(function.booleans.values())

        levels = { k: 1 for k in function.booleans }
        for (k, v) in function.integers.items():
            levels[k] = len(v) - 1

        for (var, max_level) in levels.items():
            place_map[var] = []
            for i in range(0, max_level+1):
                name = f"p{var}_b{i}"
                places[name] = (var, i)        
                place_map[var].append(name)

        if function.variable in function.booleans:
            # This is a Boolean variable.
            regulators = function.function_inputs
            var_implicants = {}
            bdd_var = function.booleans[function.variable]
            # Consider 0 -> 1 transition.
            # The restriction ensures only relevant implicants are supplied
            # while the modified variable itself does not appear in them.                        
            bdd = function.function[1].restrict({ bdd_var: False })
            implicants = []
            for bdd in expand_universal_integers(function.ctx, function.integers, regulators, bdd):                
                implicants += list_implicants(function.ctx, boolean_bdd_variables, bdd)                    
            var_implicants[(0, 1)] = implicants
            # Consider 1 -> 0 transition.
            bdd = function.function[0].restrict({ bdd_var: True })
            implicants = []
            for bdd in expand_universal_integers(function.ctx, function.integers, regulators, bdd):
                implicants += list_implicants(function.ctx, boolean_bdd_variables, bdd)
            var_implicants[(1, 0)] = implicants
            model_implicants[function.variable] = var_implicants
        else:
            # This is a multi-valued variable
            regulators = function.function_inputs
            symbolic_domain = function.integers[function.variable]
            var_implicants = {}
            if not unitary:
                # Create general (non-unitary transitions).
                for s_level in range(0, len(symbolic_domain)):
                    s_bdd_var = symbolic_domain[s_level]
                    for t_level in range(0, len(symbolic_domain)):
                        if s_level == t_level:
                            continue
                        
                        # Consider s_level -> t_level transition.
                        # Again, we want to eliminate the actual variable from the
                        # implicants while only considering relevant implicants.
                        restriction = { x: False for x in symbolic_domain }
                        restriction[s_bdd_var] = True
                        bdd = function.function[t_level].restrict(restriction)
                        implicants = []
                        for bdd in expand_universal_integers(function.ctx, function.integers, regulators, bdd):
                            implicants += list_implicants(function.ctx, boolean_bdd_variables, bdd)
                        var_implicants[(s_level, t_level)] = implicants
            else:
                # Create unitary transitions.
                for s_level in range(0, len(symbolic_domain)):
                    s_bdd_var = symbolic_domain[s_level]
                    
                    # Consider s_level -> s_level + 1 transition.
                    if s_level != len(symbolic_domain) - 1:

                        # Take a union of all implicants for larger levels.
                        bdd = function.ctx.mk_const(False)
                        for x in function.function[s_level + 1:]:
                            bdd = bdd.l_or(x)

                        # Eliminate the modified variable from the implicants.
                        restriction = { x: False for x in symbolic_domain }
                        restriction[s_bdd_var] = True
                        bdd = bdd.restrict(restriction)


                        implicants = []
                        for bdd in expand_universal_integers(function.ctx, function.integers, regulators, bdd):
                            implicants += list_implicants(function.ctx, boolean_bdd_variables, bdd)
                        var_implicants[(s_level, s_level + 1)] = implicants
                        
                    # Consider s_level -> s_level - 1 transition.
                    if s_level != 0:

                        bdd = function.ctx.mk_const(False)
                        for x in function.function[:s_level]:
                            bdd = bdd.l_or(x)
                    
                        restriction = { x: False for x in symbolic_domain }
                        restriction[s_bdd_var] = True
                        bdd = bdd.restrict(restriction)

                        implicants = []
                        for bdd in expand_universal_integers(function.ctx, function.integers, regulators, bdd):
                            implicants += list_implicants(function.ctx, boolean_bdd_variables, bdd)
                        var_implicants[(s_level, s_level - 1)] = implicants
            model_implicants[function.variable] = var_implicants  

        return Petri_Net(places, place_map, model_implicants)

    @staticmethod
    def build(model: Symbolic_Model, unitary=True) -> Petri_Net:
        places = {}
        place_map: Dict[str, List[str]] = {}
        model_implicants = {}
        boolean_bdd_variables = set(model.booleans.values())

        for (var, max_level) in model.levels.items():
            place_map[var] = []
            for i in range(0, max_level+1):
                name = f"p{var}_b{i}"
                places[name] = (var, i)        
                place_map[var].append(name)

        # First create transitions for Boolean variables. These are the same
        # regardless of unitary/general semantics.
        for var in model.booleans:    
            regulators = model.function_inputs[var] if var in model.function_inputs else []        
            var_implicants = {}
            bdd_var = model.booleans[var]
            # Consider 0 -> 1 transition.
            # The restriction ensures only relevant implicants are supplied
            # while the modified variable itself does not appear in them.                        
            bdd = model.functions[var][1].restrict({ bdd_var: False })
            implicants = []
            for bdd in expand_universal_integers(model.ctx, model.integers, regulators, bdd):                
                implicants += list_implicants(model.ctx, boolean_bdd_variables, bdd)                    
            var_implicants[(0, 1)] = implicants
            # Consider 1 -> 0 transition.
            bdd = model.functions[var][0].restrict({ bdd_var: True })
            implicants = []
            for bdd in expand_universal_integers(model.ctx, model.integers, regulators, bdd):
                implicants += list_implicants(model.ctx, boolean_bdd_variables, bdd)
            var_implicants[(1, 0)] = implicants
            model_implicants[var] = var_implicants


        if not unitary:
            # Create general (non-unitary transitions).
            for var in model.integers:
                regulators = model.function_inputs[var] if var in model.function_inputs else []
                var_implicants = {}
                for s_level in range(0, model.levels[var] + 1):
                    s_bdd_var = model.integers[var][s_level]
                    for t_level in range(0, model.levels[var] + 1):
                        if s_level == t_level:
                            continue

                        # Consider s_level -> t_level transition.
                        # Again, we want to eliminate the actual variable from the
                        # implicants while only considering relevant implicants.
                        restriction = { x: False for x in model.integers[var] }
                        restriction[s_bdd_var] = True
                        bdd = model.functions[var][t_level].restrict(restriction)
                        implicants = []
                        for bdd in expand_universal_integers(model.ctx, model.integers, regulators, bdd):
                            implicants += list_implicants(model.ctx, boolean_bdd_variables, bdd)
                        var_implicants[(s_level, t_level)] = implicants
                model_implicants[var] = var_implicants
        else:
            # Create unitary transitions.
            for var in model.integers:
                var_implicants = {}
                regulators = model.function_inputs[var] if var in model.function_inputs else []
                for s_level in range(0, model.levels[var] + 1):
                    s_bdd_var = model.integers[var][s_level]
                    
                    # Consider s_level -> s_level + 1 transition.
                    if s_level != model.levels[var]:                        
                        # Take a union of all implicants for larger levels.
                        bdd = model.ctx.mk_const(False)
                        for x in model.functions[var][s_level + 1:]:
                            bdd = bdd.l_or(x)

                        # Eliminate the modified variable from the implicants.
                        restriction = { x: False for x in model.integers[var] }
                        restriction[s_bdd_var] = True
                        bdd = bdd.restrict(restriction)


                        implicants = []
                        for bdd in expand_universal_integers(model.ctx, model.integers, regulators, bdd):
                            implicants += list_implicants(model.ctx, boolean_bdd_variables, bdd)
                        var_implicants[(s_level, s_level + 1)] = implicants
                        
                    # Consider s_level -> s_level - 1 transition.
                    if s_level > 0:
                        bdd = model.ctx.mk_const(False)
                        for x in model.functions[var][:s_level]:
                            bdd = bdd.l_or(x)
                    
                        restriction = { x: False for x in model.integers[var] }
                        restriction[s_bdd_var] = True
                        bdd = bdd.restrict(restriction)

                        implicants = []
                        for bdd in expand_universal_integers(model.ctx, model.integers, regulators, bdd):
                            implicants += list_implicants(model.ctx, boolean_bdd_variables, bdd)
                        var_implicants[(s_level, s_level - 1)] = implicants
                model_implicants[var] = var_implicants                            

        return Petri_Net(places, place_map, model_implicants)

    def merge(self, other: Petri_Net) -> Petri_Net:
        """
            Build a new `Petri_Net` which contains the variables and implicants from both arguments.
        """
        merged_places = self.places.copy()
        merged_place_map = self.place_map.copy()
        merged_implicants = self.implicants.copy()

        for (place, tuple) in other.places.items():
            if place in merged_places:
                assert tuple == merged_places[place]
            else:
                merged_places[place] = tuple
        
        for (variable, places) in other.place_map.items():
            if variable in merged_place_map:
                assert places == merged_place_map[variable]
            else:
                merged_place_map[variable] = places

        for (variable, transitions) in other.implicants.items():
            if variable in merged_implicants:
                assert transitions == merged_implicants[variable]
            else:
                merged_implicants[variable] = transitions

        return Petri_Net(merged_places, merged_place_map, merged_implicants)

    def knockout(self, variable: str) -> Petri_Net:
        """
            Create a copy of this `Petri_Net` with knockout perturbation of the given `variable` applied.
        """
        assert variable in self.place_map, f"Unknown variable {variable}."

        places = self.places.copy()
        place_map = self.place_map.copy()
        implicants = self.implicants.copy()
        ko_implicants: Dict[Tuple[int, int], List[List[str]]] = {}
        ko_max_level = len(self.place_map[variable]) - 1
        for s_level in range(1, ko_max_level + 1):
            # There is one implicant list which has no requirements, 
            # just happens any time.
            ko_implicants[(s_level, 0)] = [[]]
        implicants[variable] = ko_implicants
        return Petri_Net(places, place_map, implicants)
    
    def count_implicants(self, variable: str | None = None) -> int:
        total = 0
        for var in self.implicants:
            if variable is not None and var != variable:
                continue
            var_count = [len(x) for x in self.implicants[var].values()]
            total += sum(var_count)
        return total

def list_implicants(ctx: BddVariableSet, boolean_bdd_vars: Set[BddVariable], bdd: Bdd) -> list[list[str]]:
    """
        Return a list of positivie conjunctive clauses over the encoding variables
        which together imply the validity of this BDD.

        WARNING: Note that the BDD must not contain incorrectly encoded values.
        Use `prune_invalid` first to remove such valuations.
    """
    result = []
    for clause in bdd.clause_iterator():
        implicant = []
        for (bdd_var, value) in clause.into_list():
            if bdd_var in boolean_bdd_vars:
                # Boolean variables have to be expanded.
                if value:
                    implicant.append(ctx.name_of(bdd_var)+"_b1")
                else:
                    implicant.append(ctx.name_of(bdd_var)+"_b0")
            else:
                # Integer variables have their names baked in, but we 
                # only take the positive ones.
                if value:
                    implicant.append(ctx.name_of(bdd_var))
        result.append(implicant)
    return result

def expand_universal_integers(ctx: BddVariableSet, integers: Dict[str, List[BddVariable]], inputs: List[str], bdd: Bdd) -> list[Bdd]:
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
    for var in integers:
        if var not in inputs: 
            continue
        max_level = len(integers[var]) - 1
        new_bdds = []
        for bdd in bdds:
            shared = ctx.mk_const(True)
            for level in range(max_level + 1):
                # Variable values which guarantee this particular level.                    
                restriction = { x: False for x in integers[var] }
                restriction[integers[var][level]] = True

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