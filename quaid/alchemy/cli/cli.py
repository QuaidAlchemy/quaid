import warnings

from quaid.alchemy.cli.alchemy import alchemy
from quaid.alchemy.cli.bespoke import bespoke
from quaid.alchemy.cli.prep import prep

# filter all openfe user charge warnings in the CLI
warnings.filterwarnings(
    action="ignore",
    message="Partial charges have been provided, these will preferentially be used instead of generating new partial charges",
    category=UserWarning,
)

alchemy.add_command(prep)
alchemy.add_command(bespoke)
