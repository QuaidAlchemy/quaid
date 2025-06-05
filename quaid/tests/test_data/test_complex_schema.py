import pytest
from quaid.data.schema.complex import Complex
from quaid.data.schema.target import Target
from pydantic import ValidationError

def test_complex_from_pdb(tyk2_receptor_pdb, benzene):
    target = Target.from_pdb(
        tyk2_receptor_pdb,
        target_name="test_target",
    )

    comp = Complex(target=target, ligand=benzene)

    assert comp.target.target_name == "test_target"
    assert comp.ligand.compound_name == "benzene"
    assert comp.unique_name == "test_target-680832336d0e7a967129fb9dc89310be1b0ef61b1b8bda8b185ffa1b1437f2c3+UHOVQNZJYSORNB-UHFFFAOYNA-N"

def test_equal(tyk2_receptor_pdb, benzene):
    target = Target.from_pdb(tyk2_receptor_pdb, target_name="test_target")
    comp1 = Complex(target=target, ligand=benzene)
    comp2 = Complex(target=target, ligand=benzene)
    assert comp1 == comp2

def test_complex_json_roundtrip(tyk2_receptor_pdb, benzene):
    target = Target.from_pdb(
        tyk2_receptor_pdb,
        target_name="test_target",
    )

    comp = Complex(target=target, ligand=benzene)
    comp2 = Complex.from_json(comp.json())
    assert comp == comp2

def test_complex_json_file_roundtrip(tyk2_receptor_pdb, benzene, tmp_path):
    target = Target.from_pdb(
        tyk2_receptor_pdb,
        target_name="test_target",
    )

    comp = Complex(target=target, ligand=benzene)
    path = tmp_path / "test.json"
    comp.to_json_file(path)
    comp2 = Complex.from_json_file(path)
    assert comp == comp2
