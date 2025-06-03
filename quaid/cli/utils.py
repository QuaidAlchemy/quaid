from typing import TYPE_CHECKING, Literal

import click
import rich

if TYPE_CHECKING:
    from quaid.alchemy.schema.fec import FreeEnergyCalculationNetwork
    from quaid.data.schema.ligand import Ligand
    from cinnabar import FEMap


def print_header(console: "rich.Console"):
    """Print an ASAP-Alchemy header"""

    console.line()
    console.rule("ASAP-Alchemy")
    console.line()


def print_message(console: "rich.Console", message: str):
    """
    Print a padded message to the console using rich.

    Args:
        console: The console we should print the message to.
        message: The message to be printed.
    """
    from rich.padding import Padding

    message = Padding(message, (1, 0, 1, 0))
    console.print(message)


def has_warhead(ligand: "Ligand") -> bool:
    """
    Check if the molecule has a potential covalent warhead based on the presence of some simple SMARTS patterns.

    Args:
        ligand: The ligand which we should check for potential covalent warheads

    Returns:
        `True` if the ligand has a possible warhead else `False`.

    Notes:
        The list of possible warheads is not exhaustive and so the molecule may still be a covalent ligand.
    """
    from rdkit import Chem

    covalent_warhead_smarts = {
        "acrylamide": "[C;H2:1]=[C;H1]C(N)=O",
        "acrylamide_adduct": "NC(C[C:1]S)=O",
        "chloroacetamide": "Cl[C;H2:1]C(N)=O",
        "chloroacetamide_adduct": "S[C:1]C(N)=O",
        "vinylsulfonamide": "NS(=O)([C;H1]=[C;H2:1])=O",
        "vinylsulfonamide_adduct": "NS(=O)(C[C:1]S)=O",
        "nitrile": "N#[C:1]-[*]",
        "nitrile_adduct": "C-S-[C:1](=N)",
        "propiolamide": "NC(=O)C#C",
        "sulfamate": "NS(=O)(=O)O",
    }
    rdkit_mol = ligand.to_rdkit()
    for smarts in covalent_warhead_smarts.values():
        if rdkit_mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
            return True
    return False


class SpecialHelpOrder(click.Group):
    # from https://stackoverflow.com/questions/47972638/how-can-i-define-the-order-of-click-sub-commands-in-help
    def __init__(self, *args, **kwargs):
        self.help_priorities = {}
        super().__init__(*args, **kwargs)

    def get_help(self, ctx):
        self.list_commands = self.list_commands_for_help
        return super().get_help(ctx)

    def list_commands_for_help(self, ctx):
        """reorder the list of commands when listing the help"""
        commands = super().list_commands(ctx)
        return (
            c[1]
            for c in sorted(
                (self.help_priorities.get(command, 1), command) for command in commands
            )
        )

    def command(self, *args, **kwargs):
        """Behaves the same as `click.Group.command()` except capture
        a priority for listing command names in help.
        """
        help_priority = kwargs.pop("help_priority", 1)
        help_priorities = self.help_priorities

        def decorator(f):
            cmd = super(SpecialHelpOrder, self).command(*args, **kwargs)(f)
            help_priorities[cmd.name] = help_priority
            return cmd

        return decorator


def report_alchemize_clusters(alchemical_clusters, outsiders):
    """does some reporting alchemical cluster and outsider composition for asap-test_alchemy.prep.alchemize().
    Returns dicts that report {number-of-compounds-in-cluster : number-of-clusters-of-this-size, ..} for
    both alchemical clusters and outsider clusters. Also returns the total number of compounds in
    alchemical clusters."""
    from collections import Counter

    alchemical_cluster_sizes = dict(
        Counter([len(v) for _, v in alchemical_clusters.items()])
    )
    outsider_cluster_sizes = dict(Counter([len(v) for _, v in outsiders.items()]))

    # sort the dicts for easier interpretation of reports
    alchemical_cluster_sizes = dict(
        sorted(alchemical_cluster_sizes.items(), reverse=True)
    )
    outsider_cluster_sizes = dict(sorted(outsider_cluster_sizes.items(), reverse=True))

    alchemical_num_in_clusters = sum(
        [
            cluster_size * num_clusters
            for cluster_size, num_clusters in alchemical_cluster_sizes.items()
        ]
    )
    return alchemical_cluster_sizes, outsider_cluster_sizes, alchemical_num_in_clusters


def cinnabar_femap_is_connected(fe_map: "FEMap") -> bool:
    """Checks whether the provided femap is connected or not. Convenience function to make function
    naming clearer compared to cinnabar nomenclature."""
    return fe_map.check_weakly_connected()


def cinnabar_femap_get_largest_subnetwork(
    fe_map: "FEMap",
    result_network: "FreeEnergyCalculationNetwork",
    console: "rich.Console",
) -> "FEMap":
    """From a disconnected femap, returns the subnetwork with the largest number of nodes using a networkx
    workaround. Requires the original FreeEnergyCalculationNetwork to query results from.

    Returns a cinnabar FEMap that is fully connected"""
    import itertools

    import networkx as nx
    from quaid.alchemy.schema.fec import (
        AlchemiscaleResults,
        FreeEnergyCalculationNetwork,
    )
    from rich.padding import Padding

    fe_map_nx = fe_map.graph
    subnetworks_nodenames = sorted(  # split the network into subnetworks
        nx.strongly_connected_components(fe_map_nx), key=len, reverse=True
    )

    # ideally we'd just convert the adjust networkx back to a cinnabar fe_map but this isn't
    # implemented yet in cinnabar. Instead just take these ligands out of the result_network
    ligands_to_discard = [
        ligand for ligand in itertools.chain.from_iterable(subnetworks_nodenames[1:])
    ]

    message = Padding(
        f"Warning: removing {len(ligands_to_discard)} disconnected compounds: {round(len(ligands_to_discard)/len(fe_map_nx.nodes)*100, 2)}% of total in network. "
        f"These will not have results in the final output! Compound names: {ligands_to_discard}",
        (1, 0, 1, 0),
    )
    console.print(message)

    filtered_network_results = []
    for res in result_network.results.results:
        if (
            res.ligand_a not in ligands_to_discard
            and res.ligand_b not in ligands_to_discard
        ):
            filtered_network_results.append(res)

    # AlchemiscaleResults is immutable so need to construct a new results network with these new results
    new_results = AlchemiscaleResults(
        network_key=result_network.results.network_key, results=filtered_network_results
    )
    old_data = result_network.dict(exclude={"results"})
    new_result_network = FreeEnergyCalculationNetwork(**old_data, results=new_results)

    return new_result_network.results.to_fe_map()


def get_cpus(cpus: Literal["auto", "all"] | int) -> int:
    """
    Work out the number of cpus to use based on the request and the machine.

    Args:
        cpus: The number of cpus to use or a supported setting, "auto" or "all".

    Returns:
        The number of cpus to use.
    """
    from multiprocessing import cpu_count

    # workout the number of processes to use if auto or all
    all_cpus = cpu_count()
    if cpus == "all":
        processors = all_cpus
    elif cpus == "auto":
        processors = all_cpus - 1
    else:
        # can be a string from click
        processors = int(cpus)
    return processors
