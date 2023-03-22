from trapmvn.representation.sbml import SBML_Model
from trapmvn.core import trapmvn_async
import sys

model = sys.argv[1]
problem = sys.argv[2]
semantics = sys.argv[3]
max_output = int(sys.argv[4])
if len(sys.argv) == 6:
    fix = sys.argv[5]
else:
    fix = "deadlock"

model = SBML_Model.from_file(model)

count = 0

def count_results(trap):
    global count
    count += 1
    return max_output == 0 or count < max_output

trapmvn_async(model, count_results, semantics, problem, fix)
print(count)