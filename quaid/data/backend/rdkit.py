from pathlib import Path
from typing import Union

from quaid.data.schema.schema_base import read_file_directly
from rdkit import Chem


def _set_SD_data(mol: Union[Chem.Mol, Chem.Conformer], data: dict[str, str]):
    """
    Set the SD test_data on an rdkit molecule or conformer

    Parameters
    ----------
    mol: Union[Chem.Mol, Chem.Conformer]
        rdkit molecule or conformer

    data: dict[str, str]
        Dictionary of SD test_data to set
    """
    for key, value in data.items():
        mol.SetProp(str(key), str(value))


def _clear_SD_data(mol: Chem.Mol):
    for prop in mol.GetPropNames():
        mol.ClearProp(prop)
    # remove the name from the internal data representation
    mol.ClearProp("_Name")
    return mol


def load_sdf(file: Union[str, Path]) -> Chem.Mol:
    """
    Load an SDF file into an RDKit molecule
    """
    sdf_str = read_file_directly(file)
    return sdf_str_to_rdkit_mol(sdf_str)


def sdf_str_to_rdkit_mol(sdf: str) -> Chem.Mol:
    """
    Convert a SDF string to an RDKit molecule

    Parameters
    ----------
    sdf : str
        SDF string

    Returns
    -------
    Chem.Mol
        RDKit molecule
    """
    from io import BytesIO

    bio = BytesIO(sdf.encode())
    suppl = Chem.ForwardSDMolSupplier(bio, removeHs=False)

    ref = next(suppl)
    for mol in suppl:
        data = mol.GetPropsAsDict()
        conf = mol.GetConformer()
        _set_SD_data(conf, data)
        ref.AddConformer(conf, assignId=True)
    return ref


def rdkit_mol_to_sdf_str(mol: Chem.Mol) -> str:
    """
    Convert an RDKit molecule to a SDF string

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule

    Returns
    -------
    str
        SDF string
    """
    from io import StringIO

    sdfio = StringIO()
    w = Chem.SDWriter(sdfio)
    w.write(mol)
    w.flush()
    return sdfio.getvalue()


def rdkit_smiles_roundtrip(smi: str) -> str:
    """
    Roundtrip a SMILES string through RDKit to canonicalize it

    Parameters
    ----------
    smi : str
        SMILES string to canonicalize

    Returns
    -------
    str
        Canonicalized SMILES string
    """
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol)

def rdkit_mol_to_smiles(mol: Chem.Mol) -> str:
    """
    Convert an RDKit molecule to a SMILES string

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule

    Returns
    -------
    str
        SMILES string
    """
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

def rdkit_mol_to_inchi(mol: Chem.Mol, fixed_hydrogens: bool = False) -> str:
    """
    Convert an RDKit molecule to an InChI string

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule
    fixed_hydrogens : bool, optional
        Whether to use fixed hydrogens in the InChI string, by default False

    Returns
    -------
    str
        InChI string
    """
    inchi_options = "/LargeMolecules"
    if fixed_hydrogens:
        inchi_options += " /FixedH"
    return Chem.MolToInchi(mol, inchi_options)

def rdkit_mol_to_inchi_key(mol: Chem.Mol, fixed_hydrogens: bool = False) -> str:
    """
    Convert an RDKit molecule to an InChI Key string

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule
    fixed_hydrogens : bool, optional
        Whether to use fixed hydrogens in the InChI string, by default False

    Returns
    -------
    str
        InChI Key string
    """
    inchi_options = "/LargeMolecules"
    if fixed_hydrogens:
        inchi_options += " /FixedH"

    return Chem.MolToInchiKey(mol, inchi_options)


def rdkit_mol_from_smiles(smi: str) -> Chem.Mol:
    """
    Convert a SMILES string to an RDKit molecule

    Parameters
    ----------
    smi : str
        SMILES string

    Returns
    -------
    Chem.Mol
        RDKit molecule
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Could not convert SMILES '{smi}' to RDKit molecule.")
    # add Hs and return
    return Chem.AddHs(mol)
