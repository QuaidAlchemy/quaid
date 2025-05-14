from quaid.data.metadata.resources import seqres_data
from quaid.data.services.postera.manifold_data_validation import TargetTags


def seqres_by_target(target: TargetTags):
    resource = target.value
    try:
        res = seqres_data[resource]
    except KeyError:
        raise ValueError(f"Resource {resource} not found.")
    return res
