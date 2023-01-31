
from trapmvn.representation.sbml import SBML_Model
from trapmvn.representation.bma import BMA_Model
from trapmvn.representation.symbolic import Symbolic_Model
from trapmvn.representation.petri_net import Petri_Net

# These tests are similar to `read_model_test`, but they also build
# the symbolic representation of the model and the petri net.

def test_build_symbolic_sbml_models(sbml_file):
    model = SBML_Model.from_file(sbml_file)    
    model = Symbolic_Model.from_sbml(model)
    assert len(model.booleans) + len(model.integers) > 0
    pn = Petri_Net.build(model)
    assert len(pn.implicants) == len(model.levels)
    pn = Petri_Net.build(model, unitary=False)
    assert len(pn.implicants) == len(model.levels)


def test_build_symbolic_bma_json_models(bma_json_file):
    model = BMA_Model.from_json_file(bma_json_file)    
    model = Symbolic_Model.from_bma(model)
    assert len(model.booleans) + len(model.integers) > 0
    pn = Petri_Net.build(model)
    assert len(pn.implicants) == len(model.levels)
    pn = Petri_Net.build(model, unitary=False)
    assert len(pn.implicants) == len(model.levels)

def test_build_symbolic_bma_xml_models(bma_xml_file):
    model = BMA_Model.from_xml_file(bma_xml_file)
    model = Symbolic_Model.from_bma(model)
    assert len(model.booleans) + len(model.integers) > 0
    pn = Petri_Net.build(model)
    assert len(pn.implicants) == len(model.levels)
    pn = Petri_Net.build(model, unitary=False)
    assert len(pn.implicants) == len(model.levels)