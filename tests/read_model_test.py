
from trapmvn.representation.sbml import SBML_Model
from trapmvn.representation.bma import BMA_Model

# These tests just verify that we can "read" the test models, but 
# we have no way of verifying that we are reading them correctly.

def test_read_sbml_models(sbml_file):
    model = SBML_Model.from_file(sbml_file)
    assert len(model.variables) > 0
    assert len(model.functions) > 0

def test_read_bma_json_models(bma_json_file):
    model = BMA_Model.from_json_file(bma_json_file)
    assert len(model.variables) > 0
    assert len(model.functions) > 0

def test_read_bma_xml_models(bma_xml_file):
    model = BMA_Model.from_xml_file(bma_xml_file)
    assert len(model.variables) > 0
    assert len(model.functions) > 0
