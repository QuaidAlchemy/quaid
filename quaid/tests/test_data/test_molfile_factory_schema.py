from quaid.data.readers.molfile import MolFileFactory

def test_molfile_factory_sdf(tyk2_ligands_sdf):
    molfile = MolFileFactory(filename=tyk2_ligands_sdf)
    ligands = molfile.load()
    assert len(ligands) == 2
    assert ligands[0].compound_name == "lig_ejm_54"
    assert ligands[1].compound_name == "lig_jmc_23"
