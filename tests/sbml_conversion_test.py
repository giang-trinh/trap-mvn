

from trapmvn.representation.sbml import SBML_Model
from trapmvn.representation.bma import BMA_Model
from trapmvn.representation.symbolic import Symbolic_Model

def test_convert_bma_json_models(bma_json_file):
    model = BMA_Model.from_json_file(bma_json_file)
    symbolic = Symbolic_Model.from_bma(model)
    sbml = symbolic.to_sbml()
    sbml_xml = sbml.to_xml_string()
    sbml = SBML_Model.from_string(sbml_xml)
    assert sbml.variables == model.variables

def test_convert_bma_xml_models(bma_xml_file):
    model = BMA_Model.from_xml_file(bma_xml_file)
    symbolic = Symbolic_Model.from_bma(model)
    sbml = symbolic.to_sbml()
    sbml_xml = sbml.to_xml_string()
    sbml = SBML_Model.from_string(sbml_xml)
    assert sbml.variables == model.variables
