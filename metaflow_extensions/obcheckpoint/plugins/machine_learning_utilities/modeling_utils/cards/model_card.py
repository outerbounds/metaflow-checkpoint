import json
import time
from typing import Dict, List, Union
from ...card_utils import (
    CardRefresher,
    UpadateableTable,
)
from ...datastructures import CheckpointArtifact, ModelArtifact
from ...utils.general import unit_convert
from metaflow.cards import Markdown, Table, Artifact, VegaChart
from datetime import datetime, timedelta
from threading import Thread, Event


import json
from datetime import datetime


# Helper function to format datetime
def format_datetime(iso_str):
    # Parse the ISO 8601 string to a datetime object
    dt = datetime.fromisoformat(iso_str)
    # Format the datetime object to a more readable string
    return dt.strftime("%B %d, %Y, %H:%M:%S")


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


def _get_table_row(object):
    if object["type"] == ModelArtifact.TYPE:
        return [
            Markdown(object["model_uuid"]),
            Artifact(object["type"]),
            Markdown(format_datetime(object["created_on"])),
            Markdown(_derive_appropriate_size(object["size"])),
            Artifact(object["metadata"]),
            Artifact(object["key"]),
        ]
    elif object["type"] == CheckpointArtifact.TYPE:
        return [
            Markdown(
                "%s.%s.%s"
                % (
                    object["attempt"],
                    object["name"],
                    object["version_id"],
                )
            ),
            Artifact(object["type"]),
            Markdown(format_datetime(object["created_on"])),
            Markdown(_derive_appropriate_size(object["size"])),
            Artifact(object["metadata"]),
            Artifact(object["key"]),
        ]
    return []


def _make_loaded_models_table(loaded_models: Dict[str, Dict]):
    data = []
    headers = [
        "Artifact Name",
        "ID",
        "Artifact Type",
        "Created On",
        "Size",
        "Metadata",
        "Key",
    ]

    for key, object in loaded_models.items():
        x = _get_table_row(object)
        x.insert(0, Markdown(key))
        data.append(x)

    return UpadateableTable(
        headers=headers,
        data=data,
    )


class ModelListRefresher(CardRefresher):

    CARD_ID = "model_card"

    TABLE_HEADERS = [
        "Name",
        "Created On",
        "Size",
        "Metadata",
        # "Key",
    ]

    def __init__(
        self,
        loaded_models: Dict[str, Dict],
    ) -> None:
        self._rendered = False
        self._errored = False
        self._table = None
        self._saved_models = {}
        self._loaded_models = loaded_models

        from metaflow import current

        self.current = current

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
                "# Models \n **Task %s [Attempt:%s]**"
                % (self.current.pathspec, self.current.retry_count),
            ),
            Markdown("## Loaded Models"),
        ]
        if len(self._loaded_models) > 0:
            x.append(_make_loaded_models_table(self._loaded_models))
        else:
            x.append(Markdown("_No models loaded_"))
        # TODO : Add loaded Model Information
        return x

    def _footer_components(self):
        return []

    def on_startup(self, current_card):
        current_card.extend(self._header_components())
        current_card.extend(self._footer_components())
        current_card.refresh()

    def _make_table_objects(self, data_object):
        keys_going_in_table = []
        for model in data_object:
            _model = ModelArtifact.from_dict(model)
            if _model.key in self._saved_models:
                continue
            # TODO : We can have a Table that contains all the checkpoints
            # and each time there will be a new checkpoint, we can
            # reconstruct the table.
            self._saved_models[_model.key] = [
                Markdown(str(_model.uuid)),
                Markdown(
                    format(str(_model.created_on))
                ),  # TODO Make this a human readable date
                Markdown(
                    _derive_appropriate_size(_model.size),
                ),
                Artifact(_model.metadata),
                # Markdown(_model.key),
            ]
            keys_going_in_table.append(_model.key)
        return keys_going_in_table

    def first_time_render(self, current_card, saved_models):
        current_card.clear()
        current_card.extend(self._header_components())
        keys_going_in_table = self._make_table_objects(saved_models)
        current_card.append(Markdown("## New Models Created"))
        if len(keys_going_in_table) == 0:
            current_card.append(Markdown("_No new models created_"))
            current_card.refresh()
            return

        self._table = UpadateableTable(
            data=[self._saved_models[key] for key in keys_going_in_table],
            headers=self.TABLE_HEADERS,
        )
        current_card.append(self._table)

        current_card.extend(self._footer_components())
        current_card.refresh()
        self._rendered = True

    def data_update(self, current_card, saved_models):
        keys_going_in_table = self._make_table_objects(saved_models)
        if len(keys_going_in_table) == 0:
            return
        for key in keys_going_in_table:
            self._table.update(self._saved_models[key])

        current_card.refresh()

    def on_update(self, current_card, saved_models):
        if not self._rendered:
            self.first_time_render(current_card, saved_models)
        else:
            self.data_update(current_card, saved_models)


def _derive_appropriate_size(size):
    if unit_convert(size, "B", "MB") < 1:
        return "%s KB" % str(unit_convert(size, "B", "KB"))
    elif unit_convert(size, "B", "GB") < 1:
        return "%s MB" % str(unit_convert(size, "B", "MB"))
    else:
        return "%s GB" % str(unit_convert(size, "B", "GB"))


class ModelsCollector(Thread):
    def __init__(self, refresher: CardRefresher, interval=1):
        super().__init__()
        from metaflow import current

        self.current = current
        self._interval = interval
        self._exit_event = Event()
        self._refresher = refresher
        self.daemon = True

    def collect(self):
        return self.current.model._saved_models

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
            self.run_update()
