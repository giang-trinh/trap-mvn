import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Generator, IO, List, Optional

import networkx as nx

from sbml import sbml_file_to_petri_net
from pnml import read_pnml
from bma import bma_string_to_petri_net

def node_2_atom(name: str, value: int) -> str:
    """
    Note that it would produce an interal error if atoms start with upper letters.
    We add the prefix "p" to avoid this.
    """
    return "p" + name + "_b" + str(value)

def write_asp(one_safe_PN: nx.DiGraph, node_2_domain: dict, asp_file: IO, computation: str, fixmethod: str):
    if (computation == "fix" and fixmethod == "2"):
        for node in node_2_domain.keys():
            domain = node_2_domain[node]

            atoms = []
            for i in range(domain):
                atom = node_2_atom(node, i)
                print(
                    "{", atom, "}."
                    , file=asp_file, sep=""
                )

                atoms.append(atom)

            atom_lhs = "; ".join(atoms)
            print(
                "1 {", atom_lhs, "} 1."
                , file=asp_file
            )
        
        for node, kind in one_safe_PN.nodes(data="kind"):
            if kind == "place":
                continue
            else:  # it's a transition
                preds = list(one_safe_PN.predecessors(node))
                pred_rhs = "; ".join(preds)
                print(
                    f":- {pred_rhs}."
                    , file=asp_file
                )

    else:
        all_atoms = []

        for node in node_2_domain.keys():
            domain = node_2_domain[node]
            atoms = []

            for i in range(domain):
                atom = node_2_atom(node, i)
                print(
                    "{", atom, "}.", file=asp_file, sep=""
                )
                atoms.append(atom)
                all_atoms.append(atom)

            """conflict-freeness"""
            conf_free = "; ".join(atoms)
            print(
                f":- {conf_free}.", file=asp_file
            )

            if computation == "fix":
                conf_free = "{" + conf_free + "} >= " + str(domain - 1)
                print(
                    f"{conf_free}.", file=asp_file
                )

        if computation == "max":
            max_condition = "; ".join(all_atoms)
            print(
                f"{max_condition}.", file=asp_file
            )

        for node, kind in one_safe_PN.nodes(data="kind"):
            if kind == "place":
                continue
            else:  # it's a transition, apply siphon (if one succ is true, one pred must be true)
                preds = list(one_safe_PN.predecessors(node))
                or_preds = "; ".join(preds)
                for succ in one_safe_PN.successors(node):
                    if succ not in preds:  # optimize obvious tautologies
                        print(
                            f"{or_preds} :- {succ}.", file=asp_file
                        )
    


def solve_asp(asp_filename: str, max_output: int, time_limit: int, computation: str) -> str:
    """Run an ASP solver on program asp_file and get the solutions."""
    dom_mod = "--dom-mod=3" # minimal trap spaces and fixed points

    if computation == "max":
        dom_mod = "--dom-mod=5" # maximal trap spaces

    result = subprocess.run(
        [
            "clingo",
            str(max_output),
            "--heuristic=Domain",  # set inclusion
            "--enum-mod=domRec",
            dom_mod,
            "--outf=2",  # json output
            f"--time-limit={time_limit}",
            asp_filename,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 30 and result.returncode != 10:
        if result.returncode == 20:
            """Unsastifiable i.e., there is no fixed point or no maximal trap space"""
            return ""
        else:
            print(f"Return code from clingo: {result.returncode}")
            result.check_returncode()  # will raise CalledProcessError

    return result.stdout

def get_asp_output(
    one_safe_PN: nx.DiGraph, node_2_domain: dict, max_output: int, time_limit: int, computation: str, fixmethod: str
) -> str:
    """Generate and solve ASP file."""
    (fd, tmpname) = tempfile.mkstemp(suffix=".lp", text=True)
    
    with open(tmpname, "wt") as asp_file:
        write_asp(one_safe_PN, node_2_domain, asp_file, computation, fixmethod)
    
    solutions = solve_asp(tmpname, max_output, time_limit, computation)
    
    os.close(fd)
    os.unlink(tmpname)
    
    return solutions

def place_in_sol_fix(sol: List[str], node: str, domain: int) -> str:
    fix_value = -1

    for i in range(domain):
        if "p" + node + "_b" + str(i) in sol:
            fix_value = i
            break

    return "{" + str(fix_value) + "}"

def place_in_sol(sol: List[str], node: str, domain: int) -> str:
    value_set = list(range(domain))

    for i in range(domain):
        if "p" + node + "_b" + str(i) in sol:
            value_set.remove(i)

    return "{" + ", ".join(str(x) for x in value_set) + "}"

def solution_to_value_set(node_2_domain: dict, sol: List[str], computation: str, fixmethod: str) -> List[str]:
    """Convert a list of present places in sol, to a value set."""
    if computation == "fix" and fixmethod == "2":
        return [place_in_sol_fix(sol, node, node_2_domain[node]) for node in node_2_domain.keys()]
    else:
        return [place_in_sol(sol, node, node_2_domain[node]) for node in node_2_domain.keys()]

def get_solutions(
    asp_output: str, node_2_domain: dict, computation: str, fixmethod: str
) -> Generator[List[str], None, None]:
    """Display the ASP output back as trap-spaces."""
    solutions = json.loads(asp_output)
    yield from (
        solution_to_value_set(node_2_domain, sol["Value"], computation, fixmethod)
        for sol in solutions["Call"][0]["Witnesses"]
    )

def compute_trap_spaces(
    infile: IO,
    display: bool = False,
    max_output: int = 0,
    time_limit: int = 0,
    computation: str = "min",
    fixmethod: str = "1",
    semantics: str = "general",
) -> Generator[List[str], None, None]:
    if isinstance(infile, str):
        infile = open(infile, "r", encoding="utf-8")

    #start = time.process_time()
    if infile.name.endswith(".sbml"):
        one_safe_PN, node_2_domain = sbml_file_to_petri_net(infile, unitary=(semantics=="unitary"))
    elif infile.name.endswith(".pnml"):
        one_safe_PN, node_2_domain = read_pnml(infile, semantics)
    elif infile.name.endswith(".json"):
        model_string = infile.read()
        one_safe_PN, node_2_domain = bma_string_to_petri_net(model_string, unitary=(semantics=="unitary"))
    else:
        infile.close()
        raise ValueError("Currently limited to parsing PNML/SBML files")

    #print(f"Petri net time {time.process_time() - start:.2f}\n===")
    nodes = node_2_domain.keys()
    
    if display:
        print("\t".join(nodes))
    else:
        domains = sorted(node_2_domain.values())
        domain_count = {}

        for domain in domains:
            domain_count[domain] = 0

        for node in nodes:
            domain_count[node_2_domain[node]] += 1
	
        #print("# nodes = " + str(len(nodes)))
        #print(", ".join(str(domain) + "x" + str(domain_count[domain]) for domain in domain_count.keys()))

    solutions = get_asp_output(one_safe_PN, node_2_domain, max_output, time_limit, computation, fixmethod)

    if len(solutions) > 0:
        solutions = get_solutions(solutions, node_2_domain, computation, fixmethod)
    
        if display:
            print("\n".join("\t".join(sol) for sol in solutions))
            return
        else:
            yield from solutions
    else:
        if display:
            if computation == "fix":
                print("No fixed point")

            if computation == "max":
                print("No maximal trap space")

            return
        else:
            yield from []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--max",
        type=int,
        default=0,
        help="Maximum number of solutions (0 for all).",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        default=0,
        help="Maximum number of seconds for search (0 for no-limit).",
    )
    parser.add_argument(
        "-c",
        "--computation",
        choices=["min", "max", "fix"],
        default="min",
        type=str,
        help="Computation option.",
    )
    parser.add_argument(
        "-fm",
        "--fixmethod",
        choices=["1", "2"],
        default="1",
        type=str,
        help="Method for fixed point computation.",
    )
    parser.add_argument(
        "-s",
        "--semantics",
        choices=["general", "unitary"],
        default="general",
        type=str,
        help="Semantics for the multi-valued network.",
    )
    parser.add_argument(
        "infile",
        type=argparse.FileType("r", encoding="utf-8"),
        nargs="?",
        default=sys.stdin,
        help="sbml file",
    )
    args = parser.parse_args()

    try:
        next(compute_trap_spaces(
            args.infile,
            display=True,
            max_output=args.max,
            time_limit=args.time,
            computation=args.computation,
            fixmethod=args.fixmethod,
            semantics=args.semantics,
        ))
    except StopIteration:
        pass

if __name__ == "__main__":
    main()
