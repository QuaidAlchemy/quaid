from openff.toolkit import Molecule
from quaid.data.schema.ligand import Ligand

mol = Molecule.from_smiles("CC")
mol.name = "test"
mol.generate_conformers()
lig = Ligand.from_rdkit(mol.to_rdkit())
lig.tags["test"] = "value"
print(lig)
print(lig.to_sdf_str())

lig2 = Ligand.from_sdf_str(lig.to_sdf_str())
print(lig2)

print(lig2.data)