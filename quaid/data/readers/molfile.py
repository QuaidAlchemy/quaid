import logging
from pathlib import Path
from typing import Union

from rdkit import Chem
from quaid.data.schema.ligand import Ligand
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class MolFileFactory(BaseModel):
    """
    Factory for a loading a generic molecule file into a list of Ligand objects.
    """

    filename: Union[str, Path] = Field(..., description="Path to the molecule file")

    def load(self) -> list[Ligand]:

        ligands = []
        supplier = Chem.SDMolSupplier(self.filename, removeHs=False)
        for i, rdmol in enumerate(supplier):
            if rdmol is None:
                logger.warning(f"Skipping molecule {i} in {self.filename} due to parsing error.")
                continue
            compound_name = rdmol.GetProp("_Name") if rdmol.HasProp("_Name") else f"unknown_ligand_{i}"
            ligand = Ligand.from_rdkit(rdmol, compound_name=compound_name)
            ligands.append(ligand)
        return ligands

    @validator("filename")
    @classmethod
    def check_file_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"File {v} does not exist")
        return v
