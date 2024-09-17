import json
import time
from typing import List, Union
from ...card_utils import (
    CardDecoratorInjector,
    CardRefresher,
    AsyncPeriodicRefresher,
    LineChart,
    UpadateableTable,
)
from ...datastructures import CheckpointArtifact
from .lineage_card import (
    construct_lineage_table,
    create_checkpoint_card,
    null_card,
    format_datetime,
)
from ...utils.general import unit_convert
from metaflow.cards import Markdown, Table, Artifact, VegaChart
from datetime import datetime, timedelta
from threading import Thread, Event


import json
from datetime import datetime


def human_readable_date(date):
    return date.strftime("%Y-%m-%d %H:%M:%S")


def determine_nice_value(time_range_seconds):
    """
    Function to determine the 'nice' value based on the time range in seconds.
    """
    if time_range_seconds <= 60 * 30:  # less than 30 mins
        return "second", "%H:%M:%S"
    elif time_range_seconds <= 3600:  # less than an hour
        return "minute", "%H:%M"
    elif time_range_seconds <= 86400:  # less than a day
        return "hour", "%H:%M"
    elif time_range_seconds <= 604800:  # less than a week
        return "day", "%Y-%m-%d"
    elif time_range_seconds <= 2592000:  # less than a month
        return "week", "%Y-%m-%d"
    elif time_range_seconds <= 31536000:  # less than a year
        return "month", "%Y-%m"
    else:
        return "year", "%Y"


def generate_vega_timeline(data_objects):
    # Parse the created_on field into datetime objects
    for obj in data_objects:
        obj["created_on"] = datetime.fromisoformat(obj["created_on"])

    # Sort the data by creation time
    sorted_data = sorted(data_objects, key=lambda x: x["created_on"])

    # Get the earliest and latest timestamps
    earliest_time = sorted_data[0]["created_on"]
    latest_time = sorted_data[-1]["created_on"]

    # Calculate the time range in seconds
    time_range_seconds = (latest_time - earliest_time).total_seconds()

    # If the time range is too small, add some padding to make the domain larger
    if time_range_seconds < 10:
        earliest_time = earliest_time - timedelta(seconds=10)
        latest_time = latest_time + timedelta(seconds=10)

    # Convert to milliseconds since epoch for Vega
    min_t = earliest_time.timestamp() * 1000
    max_t = latest_time.timestamp() * 1000

    # Determine the appropriate 'nice' value based on the time range
    nice_value, date_format_for_x_axis = determine_nice_value(time_range_seconds)

    # Prepare the Vega data array for the timeline
    vega_data = [
        {
            "name_version": obj["version_id"],
            "name": obj["name"],
            "created_on": obj[
                "created_on"
            ].isoformat(),  # ISO format is directly supported by Vega
            "timestamp": obj["created_on"].timestamp()
            * 1000,  # Timestamps in milliseconds
            "description": f"Version ID: {obj['name']}.{obj['version_id']}, Created On: {human_readable_date(obj['created_on'])}",
        }
        for obj in sorted_data
    ]

    # Vega spec
    vega_spec = {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "width": 800,  # Increased the width
        "height": 200,
        "padding": 5,
        "config": {
            "text": {"font": "Helvetica, Arial, sans-serif"},
            "axis": {"labelFont": "Helvetica, Arial, sans-serif", "labelFontSize": 10},
        },
        "signals": [
            {"name": "rectWidth", "value": 40},
            {"name": "rectHeight", "value": 30},
            {"name": "rectY", "value": 55},
            {"name": "rectCenter", "init": "[rectWidth/2,rectY+rectHeight/2]"},
        ],
        "data": [
            {
                "name": "checkpoints",
                "format": {"type": "json"},
                "values": vega_data,
                "transform": [
                    {
                        "type": "collect",
                        "sort": {"field": "timestamp"},  # Use timestamp for sorting
                    }
                ],
            }
        ],
        "scales": [
            {
                "name": "xScale",
                "type": "time",  # Use time scale for proper date spacing
                "domain": [min_t, max_t],  # Use min_t and max_t for the domain
                "range": [
                    {"signal": "0"},
                    {"signal": "width"},
                ],  # Correct range using width signal
                "nice": nice_value,  # Dynamically set the 'nice' value
            },
            {
                "name": "colorScale",
                "type": "ordinal",
                "domain": {"data": "checkpoints", "field": "name"},
                "range": {"scheme": "category10"},  # Use a categorical color scheme
            },
        ],
        "axes": [
            {
                "scale": "xScale",
                "orient": "bottom",
                "format": date_format_for_x_axis,
                "labelOverlap": True,
                "labelAngle": 0,  # Angled labels for better readability
            }
        ],
        "legends": [
            {
                "fill": "colorScale",
                "title": "Checkpoints",
                "orient": "right",
                "labelFontSize": 12,
                "titleFontSize": 14,
            }
        ],
        "marks": [
            {
                "type": "rect",
                "name": "rectangles",
                "from": {"data": "checkpoints"},
                "encode": {
                    "enter": {
                        "width": {"signal": "rectWidth"},
                        "height": {"signal": "rectHeight"},
                        "x": {"signal": "scale('xScale',datum.timestamp)-rectWidth/2"},
                        "y": {"signal": "rectY"},
                        "fill": {"scale": "colorScale", "field": "name"},
                        "tooltip": {"signal": "{'Description': datum.description}"},
                    },
                    "update": {"fillOpacity": {"value": 1}},
                    "hover": {"fillOpacity": {"value": 0.5}},
                },
            },
            {
                "type": "text",
                "name": "labels",
                "from": {"data": "rectangles"},
                "encode": {
                    "enter": {
                        "text": {"signal": "datum.datum.name_version"},
                        "x": {"signal": "datum.x+rectCenter[0]"},
                        "y": {"signal": "rectCenter[1]"},
                        "align": {"value": "center"},
                        "baseline": {"value": "middle"},
                        "fontWeight": {"value": "bold"},
                        "fill": {"value": "black"},
                    }
                },
                "interactive": False,
            },
            {
                "type": "rule",
                "from": {"data": "labels"},
                "encode": {
                    "enter": {
                        "x": {"signal": "datum.x"},
                        "x2": {"signal": "datum.x"},
                        "y": {"signal": "datum.y+rectCenter[0]-5"},
                        "y2": {"signal": "height"},
                        "strokeWidth": {"value": 2},
                    }
                },
            },
        ],
    }

    return vega_spec


class CheckpointListRefresher(CardRefresher):

    CARD_ID = "task_checkpoints"

    TABLE_HEADERS = [
        "Name",
        "Created On",
        "Size",
        "Metadata",
        # "Key",
    ]

    def __init__(
        self,
        loaded_checkpoint: Union[CheckpointArtifact, None],
        lineage_stack: Union[List[CheckpointArtifact], None],
        load_policy: str,
    ) -> None:
        self._rendered = False
        self._errored = False
        self._table = None
        self._timeline_chart = None
        self._loaded_checkpoint = loaded_checkpoint
        self._lineage_stack = lineage_stack
        self._load_policy = load_policy

        from metaflow import current

        self.current = current
        self._saved_checkpoints = {}

    def on_error(self, current_card, error_message):
        if isinstance(error_message, FileNotFoundError):
            return
        if isinstance(error_message, json.JSONDecodeError):
            return
        current_card.clear()
        current_card.append(
            Markdown(
                f"## Error: {str(error_message)}",
            )
        )
        current_card.refresh()
        self._errored = True
        self._rendered = False

    def _header_components(self):
        x = [
            Markdown(
                "# Checkpoints \n **Task %s [Attempt:%s]**"
                % (self.current.pathspec, self.current.retry_count),
            )
        ]
        if self._loaded_checkpoint is not None:
            x.extend(
                create_checkpoint_card(
                    self._loaded_checkpoint, self._lineage_stack, self._load_policy
                )
            )
        else:
            x.extend(null_card(self._load_policy))
        return x

    def _footer_components(self):
        if not self._loaded_checkpoint:
            return [
                Markdown("## Lineage of Loaded Checkpoint"),
                Markdown(
                    "_no lineage found_",
                ),
            ]
        lineage_md = Markdown("## Lineage of Loaded Checkpoint")
        lineage_table = construct_lineage_table(self._lineage_stack)
        return [lineage_md, lineage_table]

    def on_startup(self, current_card):
        current_card.extend(self._header_components())
        current_card.extend(self._footer_components())
        current_card.refresh()

    def first_time_render(self, current_card, data_object, force_refresh=False):
        current_card.clear()
        current_card.extend(self._header_components())
        keys_going_in_table = self._make_table_objects(data_object)
        if len(keys_going_in_table) == 0:
            current_card.extend(
                [
                    Markdown("## Checkpoints created within the task"),
                    Markdown(
                        "_no checkpoints found_",
                    ),
                ]
            )
            current_card.extend(self._footer_components())
            current_card.refresh()
            return

        self._table = UpadateableTable(
            data=[self._saved_checkpoints[key] for key in keys_going_in_table],
            headers=self.TABLE_HEADERS,
        )

        self._timeline_chart = VegaChart(
            generate_vega_timeline(data_object), show_controls=True
        )

        current_card.append(Markdown("## Checkpoints Timeline"))
        current_card.append(self._timeline_chart)
        current_card.append(Markdown("## Checkpoints created within the task"))
        current_card.append(self._table)
        current_card.extend(self._footer_components())
        current_card.refresh(force=force_refresh)
        self._rendered = True

    def _make_table_objects(self, data_object):
        keys_going_in_table = []
        for checkpoint in data_object:
            _chckpt = CheckpointArtifact.from_dict(checkpoint)
            if _chckpt.key in self._saved_checkpoints:
                continue
            self._saved_checkpoints[_chckpt.key] = [
                Markdown(str(_chckpt.name)),
                Markdown(format_datetime(str(_chckpt.created_on))),
                Markdown(
                    _derive_appropriate_size(_chckpt.size),
                ),
                Artifact(_chckpt.metadata),
                # Markdown("```json\n%s\n```" % json.dumps(_chckpt.metadata, indent=4)),
                # Markdown(_chckpt.key),
            ]
            keys_going_in_table.append(_chckpt.key)
        return keys_going_in_table

    def data_update(self, current_card, data_object):
        keys_going_in_table = self._make_table_objects(data_object)
        if len(keys_going_in_table) == 0:
            return
        for key in keys_going_in_table:
            self._table.update(self._saved_checkpoints[key])

        self._timeline_chart.update(generate_vega_timeline(data_object))

        current_card.refresh()

    def on_update(self, current_card, data_object):
        if not self._rendered:
            self.first_time_render(current_card, data_object, force_refresh=False)
        else:
            self.data_update(current_card, data_object)

    def on_final(self, current_card, data_object):
        self._saved_checkpoints = {}
        self._rendered = False
        self.first_time_render(current_card, data_object, force_refresh=True)


def _derive_appropriate_size(size):
    if unit_convert(size, "B", "MB") < 1:
        return "%s KB" % str(unit_convert(size, "B", "KB"))
    elif unit_convert(size, "B", "GB") < 1:
        return "%s MB" % str(unit_convert(size, "B", "MB"))
    else:
        return "%s GB" % str(unit_convert(size, "B", "GB"))


class CheckpointsCollector(Thread):
    def __init__(self, refresher: CardRefresher, interval=1):
        super().__init__()
        from metaflow import current

        self.current = current
        self._interval = interval
        self._exit_event = Event()
        self._refresher = refresher

    def collect(self):
        return list(self.current.checkpoint.list(attempt=self.current.retry_count))

    def final_update(self):
        current_card = self.current.card[self._refresher.CARD_ID]
        data = self.collect()
        if len(data) == 0:
            return
        self._refresher.on_final(current_card, data)

    def run_update(self):
        current_card = self.current.card[self._refresher.CARD_ID]
        data = self.collect()
        if len(data) == 0:
            return
        self._refresher.on_update(current_card, data)

    def run(self):
        if self._refresher.CARD_ID is None:
            raise ValueError("CARD_ID must be defined")
        current_card = self.current.card[self._refresher.CARD_ID]
        self._refresher.on_startup(current_card)
        while self._exit_event.is_set() is False:
            self.run_update()
            time.sleep(self._interval)

    def stop(self):
        if not self._exit_event.is_set():
            self._exit_event.set()
            # We expose a `final_update` so that the card can be
            # called with a `force` update so that the new card
            # is rendered when the thread is stopped.
            self.final_update()
            self.join()
