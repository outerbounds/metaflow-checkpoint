from datetime import datetime
from typing import List, TYPE_CHECKING
from metaflow.cards import Table, Markdown, MetaflowCardComponent, Artifact
from metaflow.plugins.cards.card_modules.basic import DagComponent, SectionComponent

if TYPE_CHECKING:
    from ...datastructures import CheckpointArtifact as Checkpoint


# Helper function to format datetime
def format_datetime(iso_str):
    # Parse the ISO 8601 string to a datetime object
    dt = datetime.fromisoformat(iso_str)
    # Format the datetime object to a more readable string
    return dt.strftime("%B %d, %Y, %H:%M:%S")


def _current_task_info():
    from metaflow import current

    name_descrip = "%s [attempt: %s]" % (current.pathspec, current.retry_count)
    task_info = Markdown("Current Task : **%s**" % name_descrip)
    return task_info


def null_card(load_policy):

    return [
        Markdown("## Loaded Checkpoint Info [No checkpoint was loaded]"),
        Markdown(f"**Load Policy**: {load_policy}"),
    ]


# Function to create a lineage table
def construct_lineage_table(lineage):
    headers = ["Index", "Pathspec", "Created On", "Name", "metadata"]
    rows = []
    for idx, checkpoint in enumerate(lineage):
        created_on = (
            format_datetime(checkpoint.created_on) if checkpoint.created_on else "N/A"
        )
        rows.append(
            [
                str(k)
                for k in [
                    idx,
                    checkpoint.pathspec,
                    created_on,
                    checkpoint.name,
                ]
            ]
            + [Artifact(checkpoint.metadata)]
        )
    return Table(headers=headers, data=rows)


def create_checkpoint_card(
    loaded_checkpoint: "Checkpoint",
    checkpoint_lineage: List["Checkpoint"],
    load_policy: str,
) -> List[MetaflowCardComponent]:

    # Function to create a human-readable table from a checkpoint
    def checkpoint_info_table(checkpoint):
        info_keys = ["name", "url", "key", "pathspec", "attempt", "created_on"]
        rows = []
        for key in info_keys:
            value = getattr(checkpoint, key)
            if key == "created_on" and value:
                value = format_datetime(value)
            rows.append([Markdown("**%s**" % key.capitalize()), Markdown(str(value))])
        return Table(data=rows, headers=["Property", "Value"])

    # Function to create a table from metadata dictionary
    def construct_metadata_table(metadata):
        md_table = []
        for k, v in metadata.items():
            md_table.append([Markdown(f"**{k}**"), Artifact(v)])
        return Table(headers=["", ""], data=md_table)

    # Create sections of the card
    loaded_info_md = Markdown("## Loaded Checkpoint Info")
    loaded_info_table = checkpoint_info_table(
        loaded_checkpoint,
    )
    card_components = [
        loaded_info_md,
        Markdown(f"**Load Policy**: {load_policy}"),
        loaded_info_table,
    ]
    if loaded_checkpoint.metadata and len(loaded_checkpoint.metadata) > 0:
        card_components.extend(
            [
                Markdown("### Metadata"),
                construct_metadata_table(loaded_checkpoint.metadata),
            ]
        )

    # Construct and return the Metaflow Card
    return card_components


# Adjust the datetime parsing and formatting as necessary based on the actual
