import json
import logging
import warnings
from pathlib import Path
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    Dict,
    Literal,
    Optional,
    Tuple,
    Union,
)

from quaid.data.backend.rdkit import (
    rdkit_mol_to_sdf_str,
    sdf_str_to_rdkit_mol,
    _clear_SD_data,
    _set_SD_data,
    load_sdf,
    rdkit_mol_to_inchi,
    rdkit_mol_to_smiles,
    rdkit_mol_to_inchi_key,
    rdkit_mol_from_smiles
)
from quaid.data.operators.state_expanders.expansion_tag import StateExpansionTag
from quaid.data.schema.identifiers import (
    BespokeParameters,
    ChargeProvenance,
    LigandProvenance,
)
from pydantic.v1 import Field, root_validator, validator

from .schema_base import (
    DataModelAbstractBase,
    schema_dict_get_val_overload,
    write_file_directly,
)
from rdkit import Chem

if TYPE_CHECKING:
    import openfe

logger = logging.getLogger(__name__)


class InvalidLigandError(ValueError): ...  # noqa: E701


# Ligand Schema
class Ligand(DataModelAbstractBase):
    """
    Schema for a Ligand.

    Has first class serialization support for SDF files as well as the typical JSON and dictionary
    serialization.

    Note that equality comparisons are done on the chemical structure data found in the `data` field, not the other fields or the SD Tags in the original SDF
    This means you can change the other fields and still have equality, but changing the chemical structure data will change
    equality.

    You must provide either a compound_name or ids field otherwise the ligand will be invalid.

    Parameters
    ----------
    compound_name : str, optional
        Name of compound, by default None
    tags : dict[str, str], optional
        Dictionary of SD tags, by default {}
    data : str, optional, private
        Chemical structure data from the SDF file stored as a string ""
    """

    compound_name: Optional[str] = Field(None, description="Name of compound")

    provenance: LigandProvenance = Field(
        ...,
        description="Identifiers for the input state of the ligand used to ensure the sdf data is correct.",
        allow_mutation=False,
    )

    expansion_tag: Optional[StateExpansionTag] = Field(
        None,
        description="Expansion tag linking this ligand to its parent in a state expansion if needed",
    )

    charge_provenance: Optional[ChargeProvenance] = Field(
        None, description="The provenance information of the local charging method."
    )

    bespoke_parameters: Optional[BespokeParameters] = Field(
        None,
        description="The bespoke parameters for this ligand organised by interaction type.",
    )

    tags: dict[str, str] = Field(
        {},
        description="Dictionary of SD tags. "
        "If multiple conformers are present, these tags represent the first conformer.",
    )

    data: str = Field(
        ...,
        description="SDF file stored as a string to hold internal data state",
        repr=False,
    )

    @root_validator(pre=True)
    @classmethod
    def _validate_at_least_one_id(cls, v):
        ids = v.get("ids")
        compound_name = v.get("compound_name")
        # check if all the identifiers are None, sometimes when this is called from
        # already instantiated ligand we need to be able to handle a dict and instantiated class
        if compound_name is None:
            if ids is None or all(
                [v is None for v in schema_dict_get_val_overload(ids)]
            ):
                raise ValueError(
                    "At least one identifier must be provide, or compound_name must be provided"
                )
        return v

    @validator("tags")
    @classmethod
    def _validate_tags(cls, v):
        # check that tags are not reserved attribute names and format partial charges
        reser_attr_names = cls.__fields__.keys()
        for k in v.keys():
            if k in reser_attr_names:
                raise ValueError(f"Tag name {k} is a reserved attribute name")
        return v

    def __hash__(self):
        return self.json().__hash__()

    def __eq__(self, other: "Ligand") -> bool:
        return self.data_equal(other)

    def data_equal(self, other: "Ligand") -> bool:
        # Take out the header block since those aren't really important in checking
        # equality
        return "\n".join(self.data.split("\n")[2:]) == "\n".join(
            other.data.split("\n")[2:]
        )

    def to_single_conformers(self) -> list["Ligand"]:
        """
        Return a Ligand object for each conformer.
        """
        return [self.from_rdkit(conf) for conf in self.to_rdkit().GetConfs()]

    def to_rdkit(self) -> Chem.Mol:
        """
        Convert the current molecule state to an RDKit molecule including all fields as SD tags.
        """
        rdkit_mol: Chem.Mol = sdf_str_to_rdkit_mol(self.data)
        data = {}
        for key in self.__fields__.keys():
            if key not in ["data", "tags"]:
                field = getattr(self, key)
                try:
                    data[key] = field.json()
                except AttributeError:
                    if field is not None:
                        data[key] = str(getattr(self, key))
        # if we have a compound name set it as the RDKit _Name prop as well
        if self.compound_name is not None:
            data["_Name"] = self.compound_name
        # dump tags as separate items
        if self.tags is not None:
            data.update({k: v for k, v in self.tags.items()})
        # if we have partial charges set them on the atoms assuming the atom ordering is not changed
        if "atom.dprop.PartialCharge" in self.tags:
            for i, charge in enumerate(
                self.tags["atom.dprop.PartialCharge"].split(" ")
            ):
                atom = rdkit_mol.GetAtomWithIdx(i)
                atom.SetDoubleProp("PartialCharge", float(charge))

        # set the SD data on the rdkit molecule
        _set_SD_data(rdkit_mol, data)
        return rdkit_mol

    @classmethod
    def from_rdkit(cls, mol: Chem.Mol, **kwargs) -> "Ligand":
        """
        Create a Ligand from an RDKit molecule
        """
        # run some sanatisation taken from openff
        rdmol = Chem.Mol(mol)

        Chem.SanitizeMol(
            rdmol,
            (
                    Chem.SANITIZE_ALL
                    ^ Chem.SANITIZE_SETAROMATICITY
                    ^ Chem.SANITIZE_ADJUSTHS
                    ^ Chem.SANITIZE_CLEANUPCHIRALITY
            ),
        )
        Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)
        Chem.Kekulize(rdmol)
        # assign the 3d stereochemistry
        Chem.AssignStereochemistryFrom3D(rdmol)
        kwargs.pop("data", None)
        sd_tags = rdmol.GetPropsAsDict()

        for key, value in sd_tags.items():
            try:
                # check to see if we have JSON of a model field
                kwargs[key] = json.loads(value)
            except json.JSONDecodeError:
                kwargs[key] = value

        # extract all passed kwargs as a tag if it has no field in the model
        keys_to_save = [
            key for key in kwargs.keys() if key not in cls.__fields__.keys()
        ]

        tags = set()
        # some keys will not be hashable, ignore them
        for key, value in kwargs.items():
            if key in keys_to_save:
                try:
                    tags.add((key, value))
                except TypeError:
                    warnings.warn(
                        f"Tag {key} with value {value} is not hashable and will not be saved"
                    )

        kwargs["tags"] = tags

        # clean the sdf data for the internal model
        sdf_str = rdkit_mol_to_sdf_str(_clear_SD_data(rdmol))
        # create the internal LigandProvenance model
        if "provenance" not in kwargs:
            provenance = LigandProvenance(
                isomeric_smiles=rdkit_mol_to_smiles(rdmol),
                inchi=rdkit_mol_to_inchi(rdmol, fixed_hydrogens=False),
                inchi_key=rdkit_mol_to_inchi_key(rdmol, fixed_hydrogens=False),
                fixed_inchi=rdkit_mol_to_inchi(rdmol, fixed_hydrogens=True),
                fixed_inchikey=rdkit_mol_to_inchi_key(rdmol, fixed_hydrogens=True),
            )
            kwargs["provenance"] = provenance
        # check for an rdkit _Name which could be used as a compound name
        if mol.HasProp("_Name") and kwargs.get("compound_name") is None:
            kwargs["compound_name"] = mol.GetProp("_Name")

        return cls(data=sdf_str, **kwargs)


    def to_openfe(self) -> "openfe.SmallMoleculeComponent":
        """
        Convert to an openfe SmallMoleculeComponent via the rdkit interface.
        """
        import openfe

        return openfe.SmallMoleculeComponent.from_rdkit(self.to_rdkit())

    @classmethod
    def from_openfe(cls, mol: "openfe.SmallMoleculeComponent", **kwargs) -> "Ligand":
        """
        Create a Ligand from an openfe SmallMoleculeComponent
        """
        return cls.from_rdkit(mol.to_rdkit(), compound_name=mol.name, **kwargs)

    @classmethod
    def from_smiles(cls, smiles: str, **kwargs) -> "Ligand":
        """
        Create a Ligand from a SMILES string
        """
        kwargs.pop("data", None)
        rdmol = rdkit_mol_from_smiles(smiles)
        return cls.from_rdkit(rdmol, **kwargs)

    @property
    def smiles(self) -> str:
        """
        Get the canonical isomeric SMILES string for the ligand
        """
        mol = self.to_rdkit()
        return rdkit_mol_to_smiles(mol)

    @property
    def inchi(self) -> str:
        """
        Get the InChI string for the ligand
        """
        mol = self.to_rdkit()
        return rdkit_mol_to_inchi(mol=mol, fixed_hydrogens=False)

    @property
    def fixed_inchi(self) -> str:
        """
        Returns
        -------
            The fixed hydrogen inchi for the ligand.
        """
        mol = self.to_rdkit()
        return rdkit_mol_to_inchi(mol=mol, fixed_hydrogens=True)

    @property
    def inchikey(self) -> str:
        """
        Get the InChIKey string for the ligand
        """
        mol = self.to_rdkit()
        return rdkit_mol_to_inchi_key(mol=mol, fixed_hydrogens=False)

    @property
    def fixed_inchikey(self) -> str:
        """
        Returns
        -------
         The fixed hydrogen layer inchi key for the ligand
        """
        mol = self.to_rdkit()
        return rdkit_mol_to_inchi_key(mol=mol, fixed_hydrogens=True)

    @classmethod
    def from_mol2(
        cls,
        mol2_file: Union[str, Path],
        **kwargs,
    ) -> "Ligand":
        """
        Read in a ligand from an MOL2 file extracting all possible SD data into internal fields.

        Parameters
        ----------
        mol2_file : Union[str, Path]
            Path to the MOL2 file
        """

        rdmol = Chem.MolFromMol2File(mol2_file)
        return cls.from_rdkit(rdmol, **kwargs)

    @classmethod
    def from_sdf(
        cls,
        sdf_file: Union[str, Path],
        **kwargs,
    ) -> "Ligand":
        """
        Read in a ligand from an SDF file extracting all possible SD data into internal fields.

        Parameters
        ----------
        sdf_file : Union[str, Path]
            Path to the SDF file
        """
        rdmol = load_sdf(sdf_file)
        return cls.from_rdkit(rdmol, **kwargs)

    @classmethod
    def from_sdf_str(cls, sdf_str: str, **kwargs) -> "Ligand":
        """
        Create a Ligand from an SDF string
        """
        kwargs.pop("data", None)
        rdmol = sdf_str_to_rdkit_mol(sdf_str)
        return cls.from_rdkit(rdmol, **kwargs)

    def to_sdf(self, filename: Union[str, Path], allow_append=False) -> None:
        """
        Write out the ligand to an SDF file with all attributes stored as SD tags

        Parameters
        ----------
        filename : Union[str, Path]
            Path to the SDF file
        allow_append : bool, optional
            Allow appending to the file, by default False

        """
        if allow_append:
            fmode = "a"
        else:
            fmode = "w"
        write_file_directly(filename, self.to_sdf_str(), mode=fmode)

    def to_sdf_str(self) -> str:
        """
        Set the SD data for a ligand to a string representation of the data
        that can be written out to an SDF file
        """
        mol = self.to_rdkit()
        return rdkit_mol_to_sdf_str(mol)

    def print_SD_data(self) -> None:
        """
        Print the SD data for the ligand
        """
        print(self.tags)

    def clear_SD_data(self) -> None:
        """
        Clear the SD data for the ligand
        """
        self.tags = {}

    def set_expansion(
        self,
        parent: "Ligand",
        provenance: dict[str, Any],
    ) -> None:
        """
        Set the expansion of the ligand with a reference to the parent ligand and the settings used to create the
        expansion.

        Parameters
        ----------
            parent: The parent ligand from which this child was created.
            provenance: The provenance dictionary of the state expander used to create this ligand created via
            `expander.provenance()` where the keys are fields of the expander and the values capture the
            associated settings.
        """
        self.expansion_tag = StateExpansionTag.from_parent(
            parent=parent, provenance=provenance
        )


def write_ligands_to_multi_sdf(
    sdf_name: Union[str, Path],
    ligands: list[Ligand],
    overwrite=False,
):
    """
    Dumb way to do this, but just write out each ligand to the same.
    Alternate way would be to flush each to OEMol and then write out
    using OE but seems convoluted.

    Note that this will overwrite the file if it exists unless overwrite is set to False

    Parameters
    ----------
    sdf_name : Union[str, Path]
        Path to the SDF file
    ligands : list[Ligand]
        List of ligands to write out
    overwrite : bool, optional
        Overwrite the file if it exists, by default False

    Raises
    ------
    FileExistsError
        If the file exists and overwrite is False
    ValueError
        If the sdf_name does not end in .sdf
    """

    sdf_file = Path(sdf_name)
    if sdf_file.exists() and not overwrite:
        raise FileExistsError(f"{sdf_file} exists and overwrite is False")

    elif sdf_file.exists() and overwrite:
        sdf_file.unlink()

    if not sdf_file.suffix == ".sdf":
        raise ValueError("SDF name must end in .sdf")

    for ligand in ligands:
        ligand.to_sdf(sdf_file, allow_append=True)
