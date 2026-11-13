"""Names for the 8 harmonized foreground channels.

Edit if configs/label_map_multiprotocol.csv's channel assignments change.
"""

import re

CHANNEL_NAMES: dict[int, str] = {
    1: "CSF",
    2: "Cortical Grey Matter",
    3: "White Matter",
    4: "ventricles",
    5: "cerebellum",
    6: "deep grey matter",
    7: "brainstem and spinal cord",
    8: "hippocampi and amygdala",
}


def _slugify(name: str) -> str:
    """Lowercase, underscore-separated version of a structure name — safe as an
    R/pandas column-name suffix (no spaces, no punctuation)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# e.g. CHANNEL_SLUGS[4] == "ventricles", CHANNEL_SLUGS[7] == "brainstem_and_spinal_cord"
CHANNEL_SLUGS: dict[int, str] = {c: _slugify(name) for c, name in CHANNEL_NAMES.items()}
