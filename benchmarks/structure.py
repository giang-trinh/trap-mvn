from trapmvn.representation.petri_net import Petri_Net
from trapmvn.representation.sbml import SBML_Model
from trapmvn.core import _optimized_translation
from trappist.bnet import read_bnet
import sys

model = SBML_Model.from_file(sys.argv[1])
bnet_file = infile = open(sys.argv[1].replace(".sbml", ".bnet"), "r", encoding="utf-8")

general_pn = _optimized_translation(model, unitary=False)
unitary_pn = _optimized_translation(model, unitary=True)
booleanized_pn = read_bnet(bnet_file)

total_booleanized = 0
for node, kind in booleanized_pn.nodes(data="kind"):
    if kind == "transition":
        total_booleanized += 1

inputs = []
for var in model.variables:
    if unitary_pn.count_implicants(var) == 0:
        inputs.append(var)

print("Variables:", len(model.variables))
print("Inputs:", len(inputs), sorted([model.variables[i] for i in inputs]))
print("Domains:", sum([model.variables[v] + 1 for v in model.variables]))
print("General implicants:", general_pn.count_implicants())
print("Unitary implicants:", unitary_pn.count_implicants())
print("Booleanized implicants:", total_booleanized)
