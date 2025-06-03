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
from quaid.pose_generation.base import _BasicConstrainedPoseGenerator
from typing import Literal, Any, Optional
from pydantic.v1 import PositiveInt, Field, PositiveFloat


class OpenEyeConstrainedPoseGenerator(_BasicConstrainedPoseGenerator):
    type: Literal["OpenEyeConstrainedPoseGenerator"] = "OpenEyeConstrainedPoseGenerator"
    max_confs: PositiveInt = Field(
        1000, description="The maximum number of conformers to try and generate."
    )
    energy_window: PositiveFloat = Field(
        20,
        description="Sets the maximum allowable energy difference between the lowest and the highest energy conformers,"
        " in units of kcal/mol.",
    )

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if the OpenEye tools are available in the current environment.
        This checks for the presence of oechem, oeomega, oedocking, and oeff.
        """
        try:
            import openeye.oechem as oechem
            import openeye.oeomega as oeomega
            import openeye.oedocking as oedocking
            import openeye.oeff as oeff
            return True
        except ImportError:
            return False

    def provenance(self) -> dict[str, Any]:
        return {
            "oechem": oechem.OEChemGetVersion(),
            "oeomega": oeomega.OEOmegaGetVersion(),
            "oedocking": oedocking.OEDockingGetVersion(),
            "oeff": oeff.OEFFGetVersion(),
        }

    def _generate_core_fragment(
        self, reference_ligand: oechem.OEMol, core_smarts: str
    ) -> oechem.OEGraphMol:
        """
        Generate an openeye GraphMol of the core fragment made from the MCS match between the ligand and core smarts
        which will be used to constrain the geometries of the ligands during pose generation.

        Parameters
        ----------
        reference_ligand: The ligand whose pose we will be constrained to match.
        core_smarts: The SMARTS pattern used to identify the MCS in the reference ligand.

        Returns
        -------
            An OEGraphMol of the MCS matched core fragment.
        """

        # work with a copy and remove the hydrogens from the reference as we dont want to constrain to them
        input_mol = oechem.OEMol(reference_ligand)
        oechem.OESuppressHydrogens(input_mol)
        # build a query mol which allows for wild card matches
        # <https://github.com/choderalab/asapdiscovery/pull/430#issuecomment-1702360130>
        smarts_mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(smarts_mol, core_smarts)
        pattern_query = oechem.OEQMol(smarts_mol)
        atomexpr = oechem.OEExprOpts_DefaultAtoms
        bondexpr = oechem.OEExprOpts_DefaultBonds
        pattern_query.BuildExpressions(atomexpr, bondexpr)
        ss = oechem.OESubSearch(pattern_query)
        oechem.OEPrepareSearch(input_mol, ss)
        core_fragment = None

        for match in ss.Match(input_mol):
            core_fragment = oechem.OEGraphMol()
            oechem.OESubsetMol(core_fragment, match)
            break

        if core_fragment is None:
            raise RuntimeError(
                f"A core fragment could not be extracted from the reference ligand using core smarts {core_smarts}"
            )
        return core_fragment

    def _generate_omega_instance(
        self, core_fragment: oechem.OEGraphMol, use_mcs: bool
    ) -> oeomega.OEOmega:
        """
        Create an instance of omega for constrained pose generation using the input core molecule and the runtime
        settings.

        Parameters
        ----------
        core_fragment: The OEGraphMol which should be used to define the constrained atoms during generation.
        use_mcs: If the core fragment is not defined by the user try and mcs match between it and the target ligands.

        Returns
        -------
            An instance of omega configured for the current run.
        """

        # Create an Omega instance
        omega_opts = oeomega.OEOmegaOptions(oeomega.OEOmegaSampling_Dense)
        # Set the fixed reference molecule
        omega_fix_opts = oeomega.OEConfFixOptions()
        omega_fix_opts.SetFixMaxMatch(10)  # allow multiple MCSS matches
        omega_fix_opts.SetFixDeleteH(True)  # only use heavy atoms
        omega_fix_opts.SetFixMol(core_fragment)  # Provide the reference ligand
        if use_mcs:
            omega_fix_opts.SetFixMCS(True)
        omega_fix_opts.SetFixRMS(
            1.0
        )  # The maximum distance between two atoms which is considered identical
        # set the matching atom and bond expressions
        atomexpr = (
            oechem.OEExprOpts_Aromaticity
            | oechem.OEExprOpts_AtomicNumber
            | oechem.OEExprOpts_RingMember
        )
        bondexpr = oechem.OEExprOpts_Aromaticity | oechem.OEExprOpts_RingMember
        omega_fix_opts.SetAtomExpr(atomexpr)
        omega_fix_opts.SetBondExpr(bondexpr)
        omega_opts.SetConfFixOptions(omega_fix_opts)
        # set the builder options
        mol_builder_opts = oeomega.OEMolBuilderOptions()
        mol_builder_opts.SetStrictAtomTypes(
            False
        )  # don't give up if MMFF types are not found
        omega_opts.SetMolBuilderOptions(mol_builder_opts)
        omega_opts.SetWarts(False)  # expand molecule title
        omega_opts.SetStrictStereo(True)  # set strict stereochemistry
        omega_opts.SetIncludeInput(False)  # don't include input
        omega_opts.SetMaxConfs(self.max_confs)  # generate lots of conformers
        omega_opts.SetEnergyWindow(self.energy_window)  # allow high energies
        omega_generator = oeomega.OEOmega(omega_opts)

        return omega_generator

    def _generate_pose(
        self,
        target_ligand: oechem.OEMol,
        reference_ligand: oechem.OEMol,
        core_smarts: Optional[str] = None,
    ) -> oechem.OEMol:
        """
        Use the configured openeye Omega instance to generate conformers for the target ligand.

        Args:
            target_ligand: The target ligand we want to generate the conformers for.
            reference_ligand: The ligand which should be used to restrain the target ligand conformers.
            core_smarts: The smarts which should be used to identify the mcs if not provided it will be determined automatically.

        Returns:
            The openeye molecule containing the posed conformers.
        """
        from quaid.data.backend.openeye import get_SD_data, set_SD_data

        if core_smarts is not None:
            core_fragment = self._generate_core_fragment(
                reference_ligand=reference_ligand, core_smarts=core_smarts
            )
            use_mcs = False
        else:
            # use the reference ligand and let openeye find the mcs match
            core_fragment = reference_ligand
            use_mcs = True

        # build and configure omega
        omega_generator = self._generate_omega_instance(
            core_fragment=core_fragment, use_mcs=use_mcs
        )

        # Get SD test_data because the omega code will silently move it to the high level
        # and that is inconsistent with what we do elsewhere
        sd_data = get_SD_data(target_ligand)

        # run omega
        return_code = omega_generator.Build(target_ligand)

        # deal with strange hydrogen DO NOT REMOVE
        oechem.OESuppressHydrogens(target_ligand)
        oechem.OEAddExplicitHydrogens(target_ligand)

        # add SD test_data back
        set_SD_data(target_ligand, sd_data)

        if (target_ligand.GetDimension() != 3) or (
            return_code != oeomega.OEOmegaReturnCode_Success
        ):
            # add the failure message as an SD tag, should be able to see visually if the molecule is 2D

            set_SD_data(
                mol=target_ligand,
                data={"omega_return_code": oeomega.OEGetOmegaError(return_code)},
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
        Use openeye oeomega to generate constrained poses for the input ligands. The core smarts is used to decide
        which atoms should be constrained if not supplied the MCS will be found by openeye.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tqdm import tqdm

        # Make oechem be quiet
        oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Quiet)

        # grab the reference ligand
        reference_ligand = prepared_complex.ligand.to_oemol()

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
                            "target_ligand": mol.to_oemol(),
                            "core_smarts": core_smarts,
                            "reference_ligand": reference_ligand,
                        },
                    )
                    for mol in ligands
                ]
                for work in as_completed(work_list):
                    target_ligand = work.result()
                    # check if coordinates could be generated
                    if "omega_return_code" in get_SD_data(target_ligand):
                        failed_ligands.append(target_ligand)
                    else:
                        result_ligands.append(target_ligand)
                    progressbar.update(1)
        else:
            for mol in tqdm(ligands, total=len(ligands)):
                posed_ligand = self._generate_pose(
                    target_ligand=mol.to_oemol(),
                    core_smarts=core_smarts,
                    reference_ligand=reference_ligand,
                )
                # check if coordinates could be generated
                if "omega_return_code" in get_SD_data(posed_ligand):
                    failed_ligands.append(posed_ligand)
                else:
                    result_ligands.append(posed_ligand)

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