import pytest

from quaid.data.schema.identifiers import LigandProvenance
from quaid.data.schema.ligand import Ligand

def test_ligand_from_smiles(propane_smiles):
    lig = Ligand.from_smiles(propane_smiles, compound_name="test_name")
    assert lig.smiles == "[H]C([H])([H])C([H])([H])C([H])([H])[H]"
    print(lig)

def test_from_smiles_ids_made(propane_smiles):
    """Make sure the ligand provenance is automatically generated."""
    lig = Ligand.from_smiles(propane_smiles, compound_name="test_name")
    assert lig.provenance.isomeric_smiles == "[H]C([H])([H])C([H])([H])C([H])([H])[H]"
    # make sure hydrogens are added to the molecule
    rdmol = lig.to_rdkit()
    assert rdmol.GetNumAtoms() == 11

def test_dict_round_trip(roundtrip_ligand):
    lig_2 = Ligand.from_dict(roundtrip_ligand.dict())
    assert roundtrip_ligand.dict() == lig_2.dict()

def test_sdf_str_round_trip(roundtrip_ligand):
    lig_2 = Ligand.from_sdf_str(roundtrip_ligand.to_sdf_str())
    assert roundtrip_ligand.dict() == lig_2.dict()

def test_sdf_file_round_trip(roundtrip_ligand, tmp_path):
    f_name = tmp_path / "ligand.sdf"
    roundtrip_ligand.to_sdf(filename=f_name)
    lig_2 = Ligand.from_sdf(f_name)
    assert roundtrip_ligand.dict() == lig_2.dict()

def test_json_roundtrip(roundtrip_ligand):
    lig_2 = Ligand.from_json(roundtrip_ligand.json())
    assert roundtrip_ligand.dict() == lig_2.dict()

def test_json_file_round_trip(roundtrip_ligand, tmp_path):
    f_name = tmp_path / "ligand.json"
    roundtrip_ligand.to_json_file(f_name)
    lig_2 = Ligand.from_json_file(f_name)
    assert roundtrip_ligand.dict() == lig_2.dict()

def test_ligand_from_smiles_hashable(propane_smiles):
    lig1 = Ligand.from_smiles(propane_smiles, compound_name="test_name")
    lig2 = Ligand.from_smiles(propane_smiles, compound_name="test_name")
    lig3 = Ligand.from_smiles(propane_smiles, compound_name="test_name")

    assert len({lig1, lig2, lig3}) == 1


def test_ligand_from_sdf(benzene_sdf):
    lig = Ligand.from_sdf(benzene_sdf, compound_name="test_name")
    assert (
        lig.smiles == "[H]c1c([H])c([H])c([H])c([H])c1[H]"
    )
    assert lig.compound_name == "test_name"


def test_ligand_from_sdf_title_used(benzene_sdf):
    # make sure the ligand title is used as the compound ID if not set
    # important test this due to complicated skip and validation logic
    lig = Ligand.from_sdf(benzene_sdf)
    assert (
        lig.smiles == "[H]c1c([H])c([H])c([H])c([H])c1[H]"
    )
    assert lig.compound_name == "benzene"


def test_inchi(propane_smiles):
    lig = Ligand.from_smiles(propane_smiles, compound_name="test_name")
    assert lig.inchi == "InChI=1S/C3H8/c1-3-2/h3H2,1-2H3"


def test_inchi_key(propane_smiles):
    lig = Ligand.from_smiles(propane_smiles, compound_name="test_name")
    assert lig.inchikey == "ATUOYWHBWRKTHZ-UHFFFAOYSA-N"


def test_fixed_inchi():
    "Make sure a tautomer specific inchi is made when requested."
    lig = Ligand.from_smiles("c1[nH]c2c(=O)[nH]c(nc2n1)N", compound_name="test")
    assert (
        lig.fixed_inchi
        == "InChI=1/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)/f/h7,10H,6H2"
    )
    assert lig.fixed_inchi != lig.inchi

def test_fixed_inchikey():
    "Make sure a tautomer specific inchikey is made when requested."
    lig = Ligand.from_smiles("c1[nH]c2c(=O)[nH]c(nc2n1)N", compound_name="test")
    assert lig.fixed_inchikey == "UYTPUPDQBNUYGX-CQCWYMDMNA-N"
    assert lig.inchikey != lig.fixed_inchikey

def test_clear_sd_data(roundtrip_ligand):
    assert roundtrip_ligand.tags
    assert "exp DG" in roundtrip_ligand.tags
    roundtrip_ligand.clear_SD_data()
    assert not roundtrip_ligand.tags

def test_to_rdkit_sd_tags(propane_smiles):
    """Make sure we can convert to an rdkit molecule without losing any SD tags."""

    molecule = Ligand.from_smiles(smiles=propane_smiles, compound_name="testing")
    rdkit_mol = molecule.to_rdkit()
    props = rdkit_mol.GetPropsAsDict(includePrivate=True)
    # we only check the none default properties as these are what are saved
    assert molecule.compound_name == props["compound_name"]
    assert molecule.provenance == LigandProvenance.parse_raw(props["provenance"])
    # make sure the name was set when provided.
    assert molecule.compound_name == props["_Name"]

def test_partial_charge_conversion(tmp_path):
    """Make sure we can convert molecules with partial charges to other formats."""
    from gufe.components import SmallMoleculeComponent

    molecule = Ligand.from_smiles("C", compound_name="test")
    # set some fake charges
    molecule.tags["atom.dprop.PartialCharge"] = (
        "-0.10868 0.02717 0.02717 0.02717 0.02717"
    )
    molecule.charge_provenance = {
        "protocol": {"type": "OpenFF", "charge_method": "am1bcc"},
        "provenance": {"openff": 1},
    }
    # make sure the charges are set converting to rdkit on the atoms and molecule level
    rdkit_mol = molecule.to_rdkit()
    for atom in rdkit_mol.GetAtoms():
        assert atom.HasProp("PartialCharge")
    assert rdkit_mol.HasProp("atom.dprop.PartialCharge")

    # test converting to openfe
    ofe = molecule.to_openfe()
    # convert to openff and make sure the charges are found
    off_mol = ofe.to_openff()
    assert off_mol.partial_charges is not None
    for i, charge in enumerate(off_mol.partial_charges.m):
        atom = rdkit_mol.GetAtomWithIdx(i)
        assert atom.GetDoubleProp("PartialCharge") == charge

    # try a json file round trip for internal workflows
    file_name = tmp_path / "test.json"
    molecule.to_json_file(file_name)
    m2 = Ligand.from_json_file(file_name)
    assert (
        m2.tags["atom.dprop.PartialCharge"]
        == molecule.tags["atom.dprop.PartialCharge"]
    )
    assert m2.charge_provenance == molecule.charge_provenance

    # try sdf round trip
    file_name = tmp_path / "test.sdf"
    molecule.to_sdf(file_name)
    m3 = Ligand.from_sdf(file_name)
    assert (
        m3.tags["atom.dprop.PartialCharge"]
        == molecule.tags["atom.dprop.PartialCharge"]
    )
    assert m2.charge_provenance == molecule.charge_provenance

    # make sure openfe picks up the user charges from sdf
    smc = SmallMoleculeComponent.from_sdf_file(file_name)
    offmol = smc.to_openff()
    rdmol = smc.to_rdkit()
    assert offmol.partial_charges is not None
    for i, charge in enumerate(offmol.partial_charges.m):
        atom = rdmol.GetAtomWithIdx(i)
        assert atom.GetDoubleProp("PartialCharge") == charge

def test_openfe_roundtrip_charges():
    """
    Make sure we can round trip molecules to and from openfe which also have partial charges
    """
    molecule = Ligand.from_smiles("C", compound_name="test")
    # set some fake charges
    molecule.tags["atom.dprop.PartialCharge"] = (
        "-0.10868 0.02717 0.02717 0.02717 0.02717"
    )
    molecule.charge_provenance = {
        "protocol": {"type": "OpenFF", "charge_method": "am1bcc"},
        "provenance": {"openff": 1},
    }

    # test converting to openfe
    fe_mol = molecule.to_openfe()
    # now convert back
    molecule_from_fe = Ligand.from_openfe(fe_mol)
    assert molecule.charge_provenance == molecule_from_fe.charge_provenance
    assert (
        molecule.tags["atom.dprop.PartialCharge"]
        == molecule_from_fe.tags["atom.dprop.PartialCharge"]
    )
