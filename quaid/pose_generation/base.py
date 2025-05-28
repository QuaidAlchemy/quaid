from quaid.data.schema.ligand import Ligand
from quaid.data.schema.complex import Complex
from pydantic import BaseModel, Field, PositiveFloat
import abc
from enum import Enum
from typing import Literal, Any, Optional
from rdkit import Chem

class PosedLigands(BaseModel):
    """
    A results class to handle the posed and failed ligands.
    """

    posed_ligands: list[Ligand] = Field(
        [], description="A list of Ligands with a single final pose."
    )
    failed_ligands: list[Ligand] = Field(
        [], description="The list of Ligands which failed the pose generation stage."
    )

# Enums for the pose selectors
class PoseSelectionMethod(str, Enum):
    Chemgauss4 = "Chemgauss4"
    Chemgauss3 = "Chemgauss3"


class PoseEnergyMethod(str, Enum):
    MMFF = "MMFF"
    Sage = "Sage"
    Parsley = "Parsley"

class _BasicConstrainedPoseGenerator(BaseModel, abc.ABC):
    """An abstract class for other constrained pose generation methods to follow from."""

    type: Literal["_BasicConstrainedPoseGenerator"] = "_BasicConstrainedPoseGenerator"

    clash_cutoff: PositiveFloat = Field(
        2.0,
        description="The distance cutoff for which we check for clashes in Angstroms.",
    )
    selector: PoseSelectionMethod = Field(
        PoseSelectionMethod.Chemgauss3,
        description="The method which should be used to select the optimal conformer.",
    )
    backup_score: PoseEnergyMethod = Field(
        PoseEnergyMethod.Sage,
        description="If the main scoring function fails to descriminate between conformers the backup score will be used based on the internal energy of the molecule.",
    )

    class Config:
        allow_mutation = True
        arbitrary_types_allowed = True

    @abc.abstractmethod
    def provenance(self) -> dict[str, Any]:
        """Return the provenance for this pose generation method."""
        ...

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Check if the pose generation method is available."""
        ...

    def _generate_poses(
        self,
        prepared_complex: Complex,
        ligands: list[Ligand],
        core_smarts: Optional[str] = None,
        processors: int = 1,
    ) -> tuple[list[Chem.Mol], list[Chem.Mol]]:
        """The main worker method which should generate ligand poses in the receptor using the reference ligand where required."""
        ...

    def generate_poses(
        self,
        prepared_complex: Complex,
        ligands: list[Ligand],
        core_smarts: Optional[str] = None,
        processors: int = 1,
    ) -> PosedLigands:
        """
        Generate poses for the given list of molecules in the target receptor.

        Note:
            We assume all stereo and states have been expanded and checked by this point.

        Args:
            prepared_complex: The prepared receptor and reference ligand which will be used to constrain the pose of the target ligands.
            ligands: The list of ligands which require poses in the target receptor.
            core_smarts: An optional smarts string which should be used to identify the MCS between the ligand and the reference, if not
                provided the MCS will be found using RDKit to preserve chiral centers.
            processors: The number of parallel process to use when generating the conformations.

        Returns:
            A list of ligands with new poses generated and list of ligands for which we could not generate a pose.
        """

        posed_ligands, failed_ligands = self._generate_poses(
            prepared_complex=prepared_complex,
            ligands=ligands,
            core_smarts=core_smarts,
            processors=processors,
        )
        # store the results, unpacking each posed conformer to a separate molecule
        result = PosedLigands()
        for oemol in posed_ligands:
            result.posed_ligands.append(Ligand.from_oemol(oemol))

        for fail_oemol in failed_ligands:
            result.failed_ligands.append(Ligand.from_oemol(fail_oemol))
        return result

    def _prune_clashes(self, receptor: Chem.Mol, ligands: list[Chem.Mol]):
        """
        Edit the conformers on the molecules in place to remove clashes with the receptor.

        Args:
            receptor: The receptor with which we should check for clashes.
            ligands: The list of ligands with conformers to prune.

        Returns:
            The ligands with clashed conformers removed.
        """
        import numpy as np

        # setup the function to check for close neighbours
        near_nbr = oechem.OENearestNbrs(receptor, self.clash_cutoff)

        for ligand in ligands:
            if ligand.NumConfs() < 10:
                # only filter if we have more than 10 confs
                continue

            poses = []
            for conformer in ligand.GetConfs():
                clash_score = 0
                for nb in near_nbr.GetNbrs(conformer):
                    if (not nb.GetBgn().IsHydrogen()) and (
                        not nb.GetEnd().IsHydrogen()
                    ):
                        # use an exponentially decaying penalty on each distance below the cutoff not between hydrogen
                        clash_score += np.exp(
                            -0.5 * (nb.GetDist() / self.clash_cutoff) ** 2
                        )

                poses.append((clash_score, conformer))
            # eliminate the worst 50% of clashes
            poses = sorted(poses, key=lambda x: x[0])
            for _, conformer in poses[int(0.5 * len(poses)) :]:
                ligand.DeleteConf(conformer)

    def _select_best_pose(
        self, receptor: Chem.Mol, ligands: list[Chem.Mol]
    ) -> list[Chem.Mol]:
        """
        Select the best pose for each ligand in place using the selected criteria.

        TODO split into separate methods once we have more selection options

        Args:
            receptor: The receptor oedu of the receptor with the binding site defined
            ligands: The list of multi-conformer ligands for which we want to select the best pose.

        Returns:
            A list of single conformer oe molecules with the optimal pose
        """
        scorers = {
            PoseSelectionMethod.Chemgauss4: oedocking.OEScoreType_Chemgauss4,
            PoseSelectionMethod.Chemgauss3: oedocking.OEScoreType_Chemgauss3,
        }
        score = oedocking.OEScore(scorers[self.selector])
        score.Initialize(receptor)
        posed_ligands = []
        for ligand in ligands:
            poses = [
                (score.ScoreLigand(conformer), conformer)
                for conformer in ligand.GetConfs()
            ]

            # check that the scorer worked else call the backup
            # this will select the lowest energy conformer
            unique_scores = {pose[0] for pose in poses}
            if len(unique_scores) == 1 and len(poses) != 1:
                best_pose = self._select_by_energy(ligand)

            else:
                # set the best score as the active conformer
                poses = sorted(poses, key=lambda x: x[0])
                best_pose = oechem.OEGraphMol(poses[0][1])

                # set SD data to whole molecule, then get all the SD data and set to all conformers
                set_SD_data(
                    best_pose, {f"{self.selector.value}_score": str(poses[0][0])}
                )

            # turn back into a single conformer molecule
            posed_ligands.append(best_pose)
        return posed_ligands

    def _select_by_energy(self, ligand: Chem.Mol) -> Chem.Mol:
        """
        Calculate the internal energy of each conformer of the ligand using the backup score force field and select the lowest energy pose as active

        Args:
            ligand: A multi-conformer OEMol we want to calculate the energies of.

        Notes:
            This edits the molecule in place.
        """
        force_fields = {
            PoseEnergyMethod.MMFF: oeff.OEMMFF,
            PoseEnergyMethod.Sage: oeff.OESage,
            PoseEnergyMethod.Parsley: oeff.OEParsley,
        }
        ff = force_fields[self.backup_score]()
        ff.PrepMol(ligand)
        ff.Setup(ligand)
        vec_coords = oechem.OEDoubleArray(3 * ligand.GetMaxAtomIdx())
        poses = []
        for conformer in ligand.GetConfs():
            conformer.GetCoords(vec_coords)
            poses.append((ff(vec_coords), conformer))

        poses = sorted(poses, key=lambda x: x[0])
        best_pose = oechem.OEGraphMol(poses[0][1])
        set_SD_data(best_pose, {f"{self.backup_score.value}_energy": str(poses[0][0])})
        return best_pose