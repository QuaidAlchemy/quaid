from __future__ import annotations

import logging
from typing import Any

from quaid.data.schema.ligand import Ligand
from quaid.data.schema.schema_base import DataModelAbstractBase
from quaid.data.schema.target import Target
from pydantic.v1 import Field

logger = logging.getLogger(__name__)


class Complex(DataModelAbstractBase):
    """
    A complex is a combination of a Target and a Ligand.
    This is used to represent a protein-ligand complex.
    """

    target: Target = Field(description="Target schema object")
    ligand: Ligand = Field(description="Ligand schema object")

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Complex):
            return NotImplemented

        # Just check that both Targets and Ligands are the same
        return (self.target == other.target) and (self.ligand == other.ligand)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def hash(self):
        # Using the target_hash
        return f"{self.target.hash}+{self.ligand.fixed_inchikey}"

    @property
    def unique_name(self) -> str:
        """Create a unique name for the Complex, this is used in prep when generating folders to store results."""
        return f"{self.target.target_name}-{self.hash}"
