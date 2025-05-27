from typing import Any, Literal, Optional

from quaid.data.schema.schema_base import DataModelAbstractBase
from pydantic import BaseModel, Field

class LigandProvenance(DataModelAbstractBase):
    class Config:
        allow_mutation = False

    isomeric_smiles: str = Field(
        ..., description="The canonical isomeric smiles pattern for the molecule."
    )
    inchi: str = Field(..., description="The standard inchi for the input molecule.")
    inchi_key: str = Field(
        ..., description="The standard inchikey for the input molecule."
    )
    fixed_inchi: str = Field(
        ...,
        description="The non-standard fixed hydrogen layer inchi for the input molecule.",
    )
    fixed_inchikey: str = Field(
        ...,
        description="The non-standard fixed hydrogen layer inchi key for the input molecule.",
    )


class TargetIdentifiers(DataModelAbstractBase):
    """
    Identifiers for a Target
    """

    target_type: Optional[str] = Field(
        None,
        description="Tag describing the target type e.g SARS-CoV-2-Mpro, etc.",
    )

    pdb_code: Optional[str] = Field(
        None, description="The PDB code of the target if applicable"
    )


class ChargeProvenance(BaseModel):
    """A simple model to record the provenance of the local charging method."""

    class Config:
        allow_mutation = False

    type: Literal["ChargeProvenance"] = "ChargeProvenance"

    protocol: dict[str, Any] = Field(
        ..., description="The protocol and settings used to generate the local charges."
    )
    provenance: dict[str, str] = Field(
        ...,
        description="The versions of the software used to generate the local charges.",
    )


class BespokeParameter(BaseModel):
    """
    Store the bespoke parameters in a molecule.

    Note:
        This is for torsions only so far as the units are fixed to kcal / mol.
    """

    type: Literal["BespokeParameter"] = "BespokeParameter"

    interaction: str = Field(
        ..., description="The OpenFF interaction type this parameter corresponds to."
    )
    smirks: str = Field(..., description="The smirks associated with this parameter.")
    values: dict[str, float] = Field(
        {},
        description="The bespoke force field parameters "
        "which should be added to the base force field.",
    )
    units: Literal["kilocalories_per_mole"] = Field(
        "kilocalories_per_mole",
        description="The OpenFF units unit that should be attached to the values when adding the parameters "
        "to the force field.",
    )


class BespokeParameters(BaseModel):
    """A model to record the bespoke parameters for a ligand."""

    type: Literal["BespokeParameters"] = "BespokeParameters"

    parameters: list[BespokeParameter] = Field(
        [], description="The list of bespoke parameters."
    )
    base_force_field: str = Field(
        ...,
        description="The name of the base force field these parameters were "
        "derived with.",
    )
