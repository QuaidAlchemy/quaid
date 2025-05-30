from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from quaid.data.schema.ligand import Ligand


class StateExpansionTag(BaseModel):
    """
    Schema to record the expansion of a ligand state. Here we track the parent ligand and how this ligand state was
    created.

    Note we use fixed hydrogen inchikeys to distinguish between tautomers.
    """

    parent_fixed_inchikey: str = Field(
        ..., description="The fixed hydrogen map inchi key of the parent molecule."
    )
    parent_smiles: str = Field(
        ..., description="The isomeric smiles string for the parent."
    )
    provenance: dict[str, Any] = Field(
        ...,
        description="Provenance of the software used during the expansion and the state expander.",
    )

    class Config:
        allow_mutation = False

    def __hash__(self) -> int:
        return hash(self.json())

    @classmethod
    def from_parent(cls, parent: "Ligand", provenance: dict[str, str]):
        return cls(
            parent_fixed_inchikey=parent.fixed_inchikey,
            parent_smiles=parent.smiles,
            provenance=provenance,
        )
