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


def test_ligand_from_sdf(moonshot_sdf):
    lig = Ligand.from_sdf(moonshot_sdf, compound_name="test_name")
    assert (
        lig.smiles == "c1ccc2c(c1)c(cc(=O)[nH]2)C(=O)NCCOc3cc(cc(c3)Cl)O[C@H]4CC(=O)N4"
    )
    assert lig.compound_name == "test_name"


def test_ligand_from_sdf_title_used(moonshot_sdf):
    # make sure the ligand title is used as the compound ID if not set
    # important test this due to complicated skip and validation logic
    lig = Ligand.from_sdf(moonshot_sdf)
    assert (
        lig.smiles == "c1ccc2c(c1)c(cc(=O)[nH]2)C(=O)NCCOc3cc(cc(c3)Cl)O[C@H]4CC(=O)N4"
    )
    assert lig.compound_name == "Mpro-P0008_0A_ERI-UCB-ce40166b-17"


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

def test_to_rdkit_sd_tags(smiles):
    """Make sure we can convert to an rdkit molecule without losing any SD tags."""

    molecule = Ligand.from_smiles(smiles=smiles, compound_name="testing")
    rdkit_mol = molecule.to_rdkit()
    props = rdkit_mol.GetPropsAsDict(includePrivate=True)
    # we only check the none default properties as these are what are saved
    assert molecule.compound_name == props["compound_name"]
    assert molecule.provenance == LigandProvenance.parse_raw(props["provenance"])
    # make sure the name was set when provided.
    assert molecule.compound_name == props["_Name"]

def test_partial_charge_conversion(tmpdir):
    """Make sure we can convert molecules with partial charges to other formats."""
    from gufe.components import SmallMoleculeComponent

    charge_warn = "Partial charges have been provided, these will preferentially be used instead of generating new partial charges"

    with tmpdir.as_cwd():
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
        with pytest.warns(UserWarning, match=charge_warn):
            # make sure the charge warning is triggered
            ofe = molecule.to_openfe()
            # convert to openff and make sure the charges are found
            off_mol = ofe.to_openff()
            assert off_mol.partial_charges is not None
            for i, charge in enumerate(off_mol.partial_charges.m):
                atom = rdkit_mol.GetAtomWithIdx(i)
                assert atom.GetDoubleProp("PartialCharge") == charge

        # try a json file round trip for internal workflows
        molecule.to_json_file("test.json")
        m2 = Ligand.from_json_file("test.json")
        assert (
            m2.tags["atom.dprop.PartialCharge"]
            == molecule.tags["atom.dprop.PartialCharge"]
        )
        assert m2.charge_provenance == molecule.charge_provenance

        # try sdf round trip
        molecule.to_sdf("test.sdf")
        m3 = Ligand.from_sdf("test.sdf")
        assert (
            m3.tags["atom.dprop.PartialCharge"]
            == molecule.tags["atom.dprop.PartialCharge"]
        )
        assert m2.charge_provenance == molecule.charge_provenance

        # make sure openfe picks up the user charges from sdf
        with pytest.warns(UserWarning, match=charge_warn):
            _ = SmallMoleculeComponent.from_sdf_file("test.sdf")


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

    charge_warn = "Partial charges have been provided, these will preferentially be used instead of generating new partial charges"

    # test converting to openfe
    with pytest.warns(UserWarning, match=charge_warn):
        # make sure the charge warning is triggered
        fe_mol = molecule.to_openfe()
        # now convert back
        molecule_from_fe = Ligand.from_openfe(fe_mol)
        assert molecule.charge_provenance == molecule_from_fe.charge_provenance
        assert (
            molecule.tags["atom.dprop.PartialCharge"]
            == molecule_from_fe.tags["atom.dprop.PartialCharge"]
        )
