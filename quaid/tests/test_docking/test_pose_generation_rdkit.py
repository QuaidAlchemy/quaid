import pytest
from quaid.data.backend.openeye import get_SD_data
from quaid.data.schema.ligand import Ligand
from quaid.docking.pose_generation import (
    RDKitConstrainedPoseGenerator,
)
from rdkit import Chem


def test_rdkit_prov():
    """Make sure the software versions are correctly captured."""

    pose_gen = RDKitConstrainedPoseGenerator()
    provenance = pose_gen.provenance()
    assert "oechem" in provenance
    assert "oeff" in provenance
    assert "oedocking" in provenance
    assert "rdkit" in provenance
    assert "openff.toolkit" in provenance

@pytest.mark.parametrize(
    "forcefield, ff_energy",
    [
        pytest.param("MMFF", 43.42778156043702, id="MMFF"),
        pytest.param("Sage", 49.83481744522323, id="Sage"),
        pytest.param("Parsley", 128.38592742407758, id="Parsley"),
    ],
)
def test_select_by_energy(forcefield, ff_energy, mol_with_constrained_confs):
    """Test sorting the conformers by energy."""
    pose_generator = RDKitConstrainedPoseGenerator(backup_score=forcefield)
    # make sure all conformers are present
    assert 187 == mol_with_constrained_confs.NumConfs()

    # select best pose by energy
    best_pose = pose_generator._select_by_energy(ligand=mol_with_constrained_confs)
    assert mol_with_constrained_confs.GetActive().GetCoords() != best_pose.GetCoords()
    assert float(get_SD_data(best_pose)[f"{forcefield}_energy"][0]) == pytest.approx(
        ff_energy
    )



@pytest.mark.parametrize(
    "core_smarts",
    [
        pytest.param("CC1=CC2=C(CCCS2(=O)=O)C=C1", id="Core provided"),
        pytest.param(None, id="No core"),
    ],
)
def test_mcs_generate(mac1_complex, core_smarts):
    """Make sure we can generate a conformer using the mcs when we do not pass a core smarts"""

    pose_generator = RDKitConstrainedPoseGenerator()
    target_ligand = Ligand.from_smiles(
        "CCNC(=O)c1cc2c([nH]1)ncnc2N[C@@H](c3ccc4c(c3)S(=O)(=O)CCC4)C5CC5",
        compound_name="omega-error",
    )
    posed_ligands = pose_generator.generate_poses(
        prepared_complex=mac1_complex, ligands=[target_ligand], core_smarts=core_smarts
    )
    assert len(posed_ligands.posed_ligands) == 1
    # we should have no fails
    assert len(posed_ligands.failed_ligands) == 0


def test_coord_transfer_fail():
    """Make sure an error is raised if we try and transfer the coords with no matching substructure."""
    asprin = Chem.MolFromSmiles("O=C(C)Oc1ccccc1C(=O)O")
    biphenyl = Chem.MolFromSmiles("c1ccccc1-c2ccccc2")  # look for biphenyl substructure

    pose_generator = RDKitConstrainedPoseGenerator()
    with pytest.raises(RuntimeError):
        pose_generator._transfer_coordinates(
            reference_ligand=asprin, template_ligand=biphenyl
        )
