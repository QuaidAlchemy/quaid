from typing import Any, Literal, Optional

from quaid.data.backend.openeye import (
    get_SD_data,
    oechem,
    oedocking,
    oeff,
    oeomega,
    set_SD_data,
)
from quaid.data.schema.complex import Complex
from quaid.data.schema.ligand import Ligand
from pydantic.v1 import Field, PositiveFloat, PositiveInt
from rdkit import Chem, RDLogger
from quaid.pose_generation.base import _BasicConstrainedPoseGenerator

RDLogger.DisableLog(
    "rdApp.*"
)  # disables some cpp-level warnings that can break multithreading


class RDKitConstrainedPoseGenerator(_BasicConstrainedPoseGenerator):
    """Use RDKit to embed multiple conformers of the molecule while constraining it to the template ligand."""

    type: Literal["RDKitConstrainedPoseGenerator"] = "RDKitConstrainedPoseGenerator"

    max_confs: PositiveInt = Field(
        300, description="The maximum number of conformers to try and generate."
    )
    rms_thresh: PositiveFloat = Field(
        0.2,
        description="Retain only the conformations out of 'numConfs' after embedding that are at least this far apart from each other. RMSD is computed on the heavy atoms.",
    )
    mcs_timeout: PositiveInt = Field(
        1, description="The timeout in seconds to run the mcs search in RDKit."
    )

    @classmethod
    def is_available(cls) -> bool:
        # rdkit is always available if the package is installed
        return True

    def provenance(self) -> dict[str, Any]:
        import openff.toolkit
        import rdkit

        return {
            "oechem": oechem.OEChemGetVersion(),
            "oeff": oeff.OEFFGetVersion(),
            "oedocking": oedocking.OEDockingGetVersion(),
            "rdkit": rdkit.__version__,
            "openff.toolkit": openff.toolkit.__version__,
        }

    def _generate_mcs_core(
        self, target_ligand: Chem.Mol, reference_ligand: Chem.Mol
    ) -> Chem.Mol:
        """
        For the given target and reference ligand find an MCS match to generate
        a new template ligand which can be used in the constrained embedding.

        Args:
            target_ligand: The target ligand we want to generate the pose for.
            reference_ligand: The reference ligand which we want to find the mcs overlap with.

        Returns:
            An rdkit molecule created from the MCS overlap of the two ligands.
        """
        from rdkit import Chem
        from rdkit.Chem import rdFMCS

        mcs = rdFMCS.FindMCS(
            [target_ligand, reference_ligand],
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom,
            bondCompare=rdFMCS.BondCompare.CompareAny,
            maximizeBonds=False,
            timeout=self.mcs_timeout,
        )
        return Chem.MolFromSmarts(mcs.smartsString)

    def _transfer_coordinates(
        self, reference_ligand: Chem.Mol, template_ligand: Chem.Mol
    ) -> Chem.Mol:
        """
        Transfer the coordinates from the reference to the template ligand.

        Args:
            reference_ligand: The ligand we want to generate the conformers for.
            template_ligand: The ligand whose coordinates should be used as a reference.

        Returns:
            The template ligand with atom positions set to the reference for overlapping atoms.

        """
        matches = reference_ligand.GetSubstructMatch(template_ligand)
        if not matches:
            raise RuntimeError(
                f"A core fragment could not be extracted from the reference ligand using core smarts {Chem.MolToSmarts(template_ligand)}"
            )

        ref_conformer: Chem.Conformer = reference_ligand.GetConformer(0)
        template_conformer = Chem.Conformer()
        for i, atom_match in enumerate(matches):
            ref_atom_position = ref_conformer.GetAtomPosition(atom_match)
            template_conformer.SetAtomPosition(i, ref_atom_position)
        template_ligand.AddConformer(template_conformer, assignId=True)
        return template_ligand

    def _generate_coordinate_map(
        self, target_ligand: Chem.Mol, template_ligand: Chem.Mol
    ) -> tuple[dict, list]:
        """
        Generate a mapping between the target ligand atom index and the reference atoms coordinates.

        Args:
            target_ligand: The ligand we want to generate the conformers for.
            template_ligand: The ligand whose coordinates should be used as a reference.

        Returns:
            A tuple contacting a dictionary which maps the target ligand indices to a reference atom coordinate and a
            list of tuples matching the target and template ligand atom indices for any equivalent atoms.

        """
        # map the scaffold atoms to the new molecule
        # we assume the template has a single conformer
        template_conformer = template_ligand.GetConformer(0)
        match = target_ligand.GetSubstructMatch(template_ligand)
        coords_map = {}
        index_map = []
        for core_index, matched_index in enumerate(match):
            core_atom_coord = template_conformer.GetAtomPosition(core_index)
            coords_map[matched_index] = core_atom_coord
            index_map.append((matched_index, core_index))

        return coords_map, index_map

    def _generate_pose(
        self,
        target_ligand: Chem.Mol,
        core_ligand: Chem.Mol,
        core_smarts: Optional[str] = None,
    ) -> Chem.Mol:
        """
        Generate the poses for the target molecule while restraining the MCS to the core ligand.

        Args:
            target_ligand: The ligand we wish to generate the MCS restrained poses for.
            core_ligand: The reference ligand whose coordinates we should match.
            core_smarts: The smarts pattern which should be used to define the mcs between the target and the core ligand.

        Returns:
            An rdkit molecule with the generated poses to be filtered.

        Note:
            This function always returns a molecules even if generation fails it will just have no conformations.
        """

        from rdkit.Chem import AllChem  # noqa needed to trigger force fields in rdkit
        from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
        from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField
        from rdkit.Chem.rdMolAlign import AlignMol

        # run to make sure we don't lose molecule properties when using pickle
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        if core_smarts is not None:
            # extract the template mol based on this core smarts
            template_mol = self._generate_mcs_core(
                target_ligand=Chem.MolFromSmiles(core_smarts),
                reference_ligand=core_ligand,
            )
        else:
            # use mcs to find the template mol
            template_mol = self._generate_mcs_core(
                target_ligand=target_ligand, reference_ligand=core_ligand
            )
        # transfer the relevant coordinates from the crystal core to the template
        template_mol = self._transfer_coordinates(
            reference_ligand=core_ligand, template_ligand=template_mol
        )
        # create a coordinate and atom index map for the embedding
        coord_map, index_map = self._generate_coordinate_map(
            target_ligand=target_ligand, template_ligand=template_mol
        )
        # embed multiple conformers
        embeddings = list(
            EmbedMultipleConfs(
                target_ligand,
                numConfs=self.max_confs,
                clearConfs=True,
                pruneRmsThresh=self.rms_thresh,
                coordMap=coord_map,
                enforceChirality=True,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                useSmallRingTorsions=True,
            )
        )
        if len(embeddings) != 0:
            for embedding in embeddings:
                _ = AlignMol(
                    target_ligand, template_mol, prbCid=embedding, atomMap=index_map
                )

                # TODO expose MMFF as an option
                ff = UFFGetMoleculeForceField(target_ligand, confId=embedding)
                conf = template_mol.GetConformer()
                for matched_index, core_index in index_map:
                    coord = conf.GetAtomPosition(core_index)
                    coord_index = (
                        ff.AddExtraPoint(coord.x, coord.y, coord.z, fixed=True) - 1
                    )
                    ff.AddDistanceConstraint(
                        coord_index, matched_index, 0, 0, 100.0 * 100
                    )

                ff.Initialize()
                n = 4
                more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
                while more and n:
                    more = ff.Minimize(energyTol=1e-4, forceTol=1e-3)
                    n -= 1

                # realign
                _ = AlignMol(
                    target_ligand, template_mol, prbCid=embedding, atomMap=index_map
                )

        return target_ligand

    def _generate_poses(
        self,
        prepared_complex: Complex,
        ligands: list[Ligand],
        core_smarts: Optional[str] = None,
        processors: int = 1,
    ) -> tuple[list[oechem.OEMol], list[oechem.OEMol]]:
        """
        Use RDKit to embed multiple conformers which are constrained to the template molecule.

        Args:
            prepared_complex: The reference complex containing the receptor and small molecule which has been prepared.
            ligands: The list of ligands to generate poses for.
            core_smarts: The core smarts which should be used to define the core molecule.
            processors: The number of processes to use when generating the conformations.

        Returns:
            Two lists the first of the successfully posed ligands and ligands which failed.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from openff.toolkit import Molecule
        from tqdm import tqdm

        # make sure we are not using hs placed by prep as a reference coordinate for the generated conformers
        core_ligand = Chem.RemoveHs(prepared_complex.ligand.to_rdkit())

        # setup the rdkit pickle properties to save all molecule properties
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        # process the ligands
        result_ligands = []
        failed_ligands = []

        if processors > 1:
            progressbar = tqdm(total=len(ligands))
            with ProcessPoolExecutor(max_workers=processors) as pool:
                work_list = [
                    pool.submit(
                        self._generate_pose,
                        **{
                            "target_ligand": mol.to_rdkit(),
                            "core_ligand": core_ligand,
                            "core_smarts": core_smarts,
                        },
                    )
                    for mol in ligands
                ]
                for work in as_completed(work_list):
                    target_ligand = work.result()
                    off_mol = Molecule.from_rdkit(
                        target_ligand, allow_undefined_stereo=True
                    )
                    # we need to transfer the properties which would be lost
                    openeye_mol = off_mol.to_openeye()

                    # make sure properties at the top level get added to the conformers
                    sd_tags = get_SD_data(openeye_mol)
                    set_SD_data(openeye_mol, sd_tags)

                    if target_ligand.GetNumConformers() > 0:
                        # save the mol with all conformers
                        result_ligands.append(openeye_mol)
                    else:
                        failed_ligands.append(openeye_mol)

                    progressbar.update(1)
        else:
            for mol in tqdm(ligands, total=len(ligands)):
                posed_ligand = self._generate_pose(
                    target_ligand=Chem.AddHs(mol.to_rdkit()),
                    core_ligand=core_ligand,
                    core_smarts=core_smarts,
                )

                off_mol = Molecule.from_rdkit(posed_ligand, allow_undefined_stereo=True)
                # we need to transfer the properties which would be lost
                openeye_mol = off_mol.to_openeye()

                # make sure properties at the top level get added to the conformers
                sd_tags = get_SD_data(openeye_mol)
                set_SD_data(openeye_mol, sd_tags)

                if posed_ligand.GetNumConformers() > 0:
                    # save the mol with all conformers
                    result_ligands.append(openeye_mol)
                else:
                    failed_ligands.append(openeye_mol)

        # prue down the conformers
        oedu_receptor = prepared_complex.target.to_oedu()
        oe_receptor = oechem.OEGraphMol()
        oedu_receptor.GetProtein(oe_receptor)

        self._prune_clashes(receptor=oe_receptor, ligands=result_ligands)
        # select the best pose to be kept
        posed_ligands = self._select_best_pose(
            receptor=oedu_receptor, ligands=result_ligands
        )
        return posed_ligands, failed_ligands
