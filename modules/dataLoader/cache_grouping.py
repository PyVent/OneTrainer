import hashlib
import json
import math

from modules.util.loading_progress import LoadingProgress


def init_concept_groups(module, *, cached):
    """Hash settings once per concept instead of once for every image/caption pair."""
    indices, balancing, strategies, variations = {}, {}, {}, {}
    concepts = {}
    total = module._get_previous_length("concept")
    with LoadingProgress(total=total, desc="grouping cache" if cached else "grouping epoch samples", unit="example") as progress:
        for index in range(total):
            concept = module._get_previous_item(0, "concept", index)
            if concept.get("enabled", True):
                identity = id(concept)
                key = concepts.get(identity)
                if key is None:
                    data = []
                    for name in module.variations_group_in_names:
                        value = concept
                        for part in name.split(".")[1:]:
                            value = value[part]
                        data.append(value)
                    key = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
                    concepts[identity] = key
                    indices.setdefault(key, [])
                    balancing[key] = concept["balancing"]
                    strategies[key] = concept["balancing_strategy"]
                    if cached:
                        variations[key] = concept[module.variations_in_name.split(".")[-1]]
                indices[key].append(index)
            progress.update()
    module.group_indices = indices
    module.group_output_samples = {
        key: int(math.floor(len(values) * balancing[key])) if strategies[key] == "REPEATS" else int(balancing[key])
        for key, values in indices.items()
    }
    if cached:
        module.group_variations = variations
    module.aggregate_cache = {}
    module.variations_initialized = True
