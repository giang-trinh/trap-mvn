from trapmvn.representation.generator import randomly_expand_boolean_variables
from trapmvn.representation.sbml import SBML_Model
import sys

model = SBML_Model.from_file(sys.argv[1])
randomly_expand_boolean_variables(model)
print(model.to_xml_string())