import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union  # noqa: F401

from quaid.data.schema.identifiers import TargetIdentifiers
from pydantic.v1 import Field, root_validator

from quaid.data.schema.schema_base import (
    DataModelAbstractBase,
    check_strings_for_equality_with_exclusion,
    schema_dict_get_val_overload,
)
from quaid.data.schema.ligand import Ligand
from gufe.components import ProteinComponent

logger = logging.getLogger(__name__)


class InvalidTargetError(ValueError): ...  # noqa: E701


class Target(DataModelAbstractBase):
    """
    Schema for a Target.

    Wrapper around the ProteinComponent from gufe, which is used to represent a protein target with some
    ids and a name.
    """

    target_name: Optional[str] = Field(None, description="The name of the target")

    ids: Optional[TargetIdentifiers] = Field(
        None,
        description="TargetIdentifiers Schema for identifiers associated with this target",
    )

    data: str = Field(
        ...,
        description="The JSON of the openfe protein component representation of the target.",
        repr=False,
    )

    cofactors: Optional[list[Ligand]] = Field(None, description="List of cofactors associated with the target")

    @root_validator(pre=True)
    @classmethod
    def _validate_at_least_one_id(cls, v):
        # check if skip validation
        if v.get("_skip_validate_ids"):
            return v
        else:
            ids = v.get("ids")
            compound_name = v.get("target_name")
            # check if all the identifiers are None, sometimes when this is called from
            # already instantiated ligand we need to be able to handle a dict and instantiated class
            if compound_name is None:
                if ids is None or all(
                    [not v for v in schema_dict_get_val_overload(ids)]
                ):
                    raise ValueError(
                        "At least one identifier must be provide, or target_name must be provided"
                    )
        return v

    @classmethod
    def from_pdb(
        cls, pdb_file: Union[str, Path], **kwargs
    ) -> "Target":
        """
        Create a Target from a PDB file.
        """

        kwargs.pop("test_data", None)
        # directly read in test_data
        if isinstance(pdb_file, Path):
            pdb_file = pdb_file.as_posix()
        target_mol = ProteinComponent.from_pdb_file(pdb_file)
        return cls(data=target_mol.to_json(), **kwargs)

    def to_pdb(self, filename: Union[str, Path]) -> None:
        # directly write out test_data
        target_mol = ProteinComponent.from_json(content=self.data)
        target_mol.to_pdb_file(filename)

    def to_pdb_str(self) ->str:
        """
        Convert the target to a PDB string.
        """
        import io
        target_mol = ProteinComponent.from_json(content=self.data)
        pdb_str = io.StringIO()
        target_mol.to_pdb_file(pdb_str)
        return pdb_str.getvalue()


    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Target):
            return NotImplemented
        # check if the test_data is the same
        # but exclude the MASTER record as this is not always in the SAME PLACE
        # for some strange reason
        return check_strings_for_equality_with_exclusion(
            self.to_pdb_str(), other.to_pdb_str(), "MASTER"
        )

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    @property
    def hash(self):
        """Create a hash based on the pdb file contents"""
        import hashlib
        return hashlib.sha256(self.data.encode()).hexdigest()
