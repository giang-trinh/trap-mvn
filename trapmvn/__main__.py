import argparse
import sys
from trapmvn.representation.sbml import SBML_Model
from trapmvn.representation.bma import BMA_Model
from trapmvn.core import trapmvn_async
from typing import Union

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--max",
    type=int,
    default=0,
    help="Maximum number of solutions (0 for all).",
)
#parser.add_argument(
#    "-t",
#    "--time",
#    type=int,
#    default=0,
#    help="Maximum number of seconds for search (0 for no-limit).",
#)
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
    choices=["deadlock", "siphon"],
    default="deadlock",
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
    help="SBML-qual / BMA model file.",
)

args = parser.parse_args()
limit = args.max
problem = args.computation
semantics = args.semantics
fixed_point_method = args.fixmethod
infile = args.infile

model: Union[SBML_Model, BMA_Model]
if isinstance(infile, str):
    if infile.endswith(".xml") or infile.endswith("sbml"):
        model = SBML_Model.from_file(infile)
    elif infile.endswith(".json") or infile.endswith(".bma"):
        model = BMA_Model.from_json_file(infile)
    else:
        raise Exception("Unknown file format. Only SBML (XML) and BMA (JSON) files are supported.")
else:
    # "Default" behaviour is to expect SBML files.
    model = SBML_Model.from_string(infile.read())
    infile.close()

variables = sorted(model.variables.keys())

# Print header.
print('\t'.join(variables))

# Function to print individual trap spaces.
total = 0

def print_space(space):        
    global total
    space = space.decode()
    print('\t'.join([ str(space[var]) for var in variables ]))
    total += 1
    return limit == 0 or total < limit

trapmvn_async(model, print_space, semantics, problem, fixed_point_method)