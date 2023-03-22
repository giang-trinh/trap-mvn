from trapmvn.representation.sbml import SBML_Model
from trapmvn.core import trapmvn_async
from trappist.trappist import compute_trap_spaces
import sys

model = sys.argv[1]
problem = sys.argv[2]
max_output = int(sys.argv[3])
if len(sys.argv) == 5:
    fix = sys.argv[4]
else:
    fix = "1"

generator = compute_trap_spaces(
    infile = model,
    display = False,
    max_output = max_output,
    time_limit = 0,
    computation = problem,
    fixmethod = fix,
    method = "asp"
)

count = sum(1 for _ in generator)
print(count)