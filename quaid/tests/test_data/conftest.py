import pytest
from quaid.data.schema.ligand import Ligand, StateExpansionTag, ChargeProvenance, BespokeParameters
from quaid.data.schema.identifiers import BespokeParameter
from importlib import resources


@pytest.fixture(scope="session")
def propane_smiles():
    return "CCC"

@pytest.fixture(scope="session")
def roundtrip_ligand(propane_smiles):
    # a round trip ligand with all fields set
    lig = Ligand.from_smiles(propane_smiles, compound_name="propane")
    # set some tags
    lig.tags["exp DG"] = "-11.5"
    assert lig.provenance.isomeric_smiles == "[H]C([H])([H])C([H])([H])C([H])([H])[H]"
    lig.charge_provenance = ChargeProvenance(
        protocol={"type": "OpenFFCharges", "charge_method": "am1bcc"},
        provenance={"ambertools": 23, "rdkit": 1}
    )
    lig.expansion_tag = StateExpansionTag(
        parent_fixed_inchikey=lig.fixed_inchikey,
        parent_smiles=lig.smiles,
        provenance={"type": "test_expander"}
    )
    lig.bespoke_parameters = BespokeParameters(
        base_force_field="openff-2.0.0.offxml",
        parameters=[BespokeParameter(
            interaction="ProperTorsions",
            smirks="[*:1]-[#6X4:2]-[#6X4:3]-[*:4]",
            values={"k1": 0.1}
        )]
    )
    return lig

@pytest.fixture(scope="session")
def benzene_sdf():
    with resources.files("quaid.tests.data") as d:
        return d / "benzene.sdf"

@pytest.fixture(scope="session")
def tyk2_receptor_pdb():
    with resources.files("quaid.tests.data") as d:
        return d / "tyk2_receptor.pdb"

@pytest.fixture(scope="session")
def benzene(benzene_sdf) -> Ligand:
    return Ligand.from_sdf(benzene_sdf)

@pytest.fixture(scope="session")
def tyk2_ligands_sdf():
    with resources.files("quaid.tests.data") as d:
        return d / "tyk2_ligands.sdf"