import pytest
from quaid.data.schema.target import Target, TargetIdentifiers


def test_target_from_pdb_at_least_one_id(tyk2_receptor_pdb):
    with pytest.raises(ValueError):
        # neither id is set
        Target.from_pdb(tyk2_receptor_pdb)

def test_target_from_pdb_at_least_one_target_id(tyk2_receptor_pdb):
    with pytest.raises(ValueError):
        # neither id is set
        Target.from_pdb(tyk2_receptor_pdb, ids=TargetIdentifiers())

def test_target_identifiers():
    target_type = "MERS-CoV-Mpro"
    ids = TargetIdentifiers(target_type=target_type, pdb_code="blah")
    assert ids.target_type== target_type
    assert ids.pdb_code == "blah"

def test_target_identifiers_json_file_roundtrip(tmp_path):
    target_type = "MERS-CoV-Mpro"
    ids = TargetIdentifiers(target_type=target_type, pdb_code="blah")
    file_name = tmp_path / "test.json"
    ids.to_json_file(file_name)
    ids2 = TargetIdentifiers.from_json_file(file_name)
    assert ids2.target_type == target_type
    assert ids2.pdb_code == "blah"

def test_target_dict_roundtrip(tyk2_receptor_pdb):
    target_type = "SARS-CoV-2-Mpro"
    t1 = Target.from_pdb(
        tyk2_receptor_pdb,
        target_name=target_type,
        ids=TargetIdentifiers(
            target_type=target_type, pdb_code="blah"
        ),
    )
    t2 = Target.from_dict(t1.dict())
    assert t1 == t2

def test_target_json_roundtrip(tyk2_receptor_pdb):
    target_name = "SARS-CoV-2-Mpro"
    t1 = Target.from_pdb(
        tyk2_receptor_pdb,
        target_name=target_name,
    )
    t2 = Target.from_json(t1.json())
    assert t1 == t2

def test_target_json_file_roundtrip(tyk2_receptor_pdb, tmp_path):
    target_name = "SARS-CoV-2-Mpro"
    t1 = Target.from_pdb(
        tyk2_receptor_pdb,
        target_name=target_name,
        ids=TargetIdentifiers(
            target_type=target_name, pdb_code="blah"
        ),
    )
    path = tmp_path / "test.json"
    t1.to_json_file(path)
    t2 = Target.from_json_file(path)
    assert t1 == t2

def test_target_data_equal(tyk2_receptor_pdb):
    t1 = Target.from_pdb(tyk2_receptor_pdb, target_name="TargetTestName")
    t2 = Target.from_pdb(tyk2_receptor_pdb, target_name="TargetTestName")
    # does the same thing as the __eq__ method
    assert t1.data_equal(t2)
    assert t1 == t2

def test_pdb_roundtrip(tyk2_receptor_pdb, tmp_path):
    target_name = "SARS-CoV-2-Mpro"
    t1 = Target.from_pdb(
        tyk2_receptor_pdb,
        target_name=target_name,
    )
    path = tmp_path / "test.pdb"
    t1.to_pdb(path)
    t2 = Target.from_pdb(path, target_name=target_name)
    assert t1 == t2

def test_target_hash(tyk2_receptor_pdb):
    target_name = "SARS-CoV-2-Mpro"
    t1 = Target.from_pdb(
        tyk2_receptor_pdb,
        target_name=target_name,
    )
    assert t1.hash == "680832336d0e7a967129fb9dc89310be1b0ef61b1b8bda8b185ffa1b1437f2c3"