from functools import partial
import json
import time
from typing import Dict, List, Union
from ...card_utils import (
    CardRefresher,
    UpadateableTable,
)
from ...utils.general import unit_convert
from metaflow.cards import Markdown, Table, Artifact
from datetime import datetime
from threading import Thread, Event


# Helper function to format datetime
def format_datetime(iso_str):
    # Parse the ISO 8601 string to a datetime object
    dt = datetime.fromisoformat(iso_str)
    # Format the datetime object to a more readable string
    return dt.strftime("%B %d, %Y, %H:%M:%S")


def _derive_appropriate_size(size):
    """Convert size in bytes to a human-readable format"""
    if size is None or size == 0:
        return "0 KB"
    if unit_convert(size, "B", "MB") < 1:
        return "%s KB" % str(round(unit_convert(size, "B", "KB"), 2))
    elif unit_convert(size, "B", "GB") < 1:
        return "%s MB" % str(round(unit_convert(size, "B", "MB"), 2))
    else:
        return "%s GB" % str(round(unit_convert(size, "B", "GB"), 2))


def _create_model_section(
    repo_id,
    model_info,
    cache_scope,
    calling_mechanism="decorator load",
    cache_hit=None,
    force_download=None,
):
    """Create a section for a single model with a vertical table"""
    components = []

    # Extract metadata
    metadata = model_info.get("metadata", {})
    repo_type = metadata.get("repo_type", "model")
    size = model_info.get("size", 0)
    created_on = model_info.get("created_on", "")
    storage_url = model_info.get("url", "N/A")
    pathspec = model_info.get("pathspec", "N/A")

    # Get HF Hub parameters from metadata
    hf_kwargs = model_info.get("hf_kwargs", metadata.get("kwargs", {}))

    # Create model heading
    components.append(Markdown(f"### 🤗 {repo_id} (`{repo_type}`)"))

    # Create vertical table data

    UnCompressedArtifact = partial(Artifact, compressed=False)
    table_data = [
        [Markdown("**Calling Mechanism**"), UnCompressedArtifact(calling_mechanism)],
        [Markdown("**Repo Type**"), UnCompressedArtifact(repo_type)],
        [Markdown("**Size**"), UnCompressedArtifact(_derive_appropriate_size(size))],
        [Markdown("**Cache Scope**"), UnCompressedArtifact(cache_scope)],
    ]

    # Add cache information if available
    if cache_hit is not None:
        cache_status = "✅ Cache Hit" if cache_hit else "⬇️ Downloaded from HF Hub"
        table_data.append(
            [Markdown("**Cache Status**"), UnCompressedArtifact(cache_status)]
        )

    if force_download is not None:
        force_dl_status = "Yes" if force_download else "No"
        table_data.append(
            [Markdown("**Force Download**"), UnCompressedArtifact(force_dl_status)]
        )

    table_data.extend(
        [
            [
                Markdown("**Created On**"),
                UnCompressedArtifact(format_datetime(created_on))
                if created_on
                else Markdown("_unknown_"),
            ],
            [Markdown("**Cached by Task**"), UnCompressedArtifact(pathspec)],
            [Markdown("**Remote URL**"), UnCompressedArtifact(storage_url)],
        ]
    )

    # Add HF Hub parameters if available
    if hf_kwargs:
        table_data.append([Markdown("**HF Hub Parameters**"), Artifact(hf_kwargs)])

    # Create the vertical table
    components.append(Table(table_data))

    components.append(Markdown("---"))

    return components


class HuggingfaceHubListRefresher(CardRefresher):
    """Card refresher for Hugging Face Hub models"""

    CARD_ID = "huggingface_hub_models"

    def __init__(
        self,
        loaded_models_data: Dict[str, Dict],
        cache_scope: str,
    ) -> None:
        self._rendered = False
        self._errored = False
        self._loaded_models_data = loaded_models_data
        self._cache_scope = cache_scope
        self._tracked_models = {}
        self._tracked_model_keys = set()

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
        """Create header components for the card"""
        x = [
            Markdown(
                "# Hugging Face Hub Models \n **Task %s [Attempt:%s]**"
                % (self.current.pathspec, self.current.retry_count),
            ),
            Markdown(f"**Cache Scope:** `{self._cache_scope}`"),
            Markdown("---"),
            Markdown("## Models Loaded via Decorator"),
        ]

        if len(self._loaded_models_data) > 0:
            for repo_id, model_info in self._loaded_models_data.items():
                x.extend(
                    _create_model_section(
                        repo_id,
                        model_info,
                        self._cache_scope,
                        calling_mechanism="decorator load",
                        cache_hit=model_info["cache_hit"],
                        force_download=model_info["force_download"],
                    ),
                )
        else:
            x.append(Markdown("_No models loaded via decorator `load=` parameter_"))
            x.append(Markdown("---"))

        return x

    def _footer_components(self):
        """Create footer components"""
        return []

    def on_startup(self, current_card):
        """Initialize the card on startup"""
        current_card.extend(self._header_components())
        current_card.extend(self._footer_components())
        current_card.refresh()

    def _make_model_sections(self, data_object):
        """Process runtime model tracking data and create sections"""
        new_components = []
        for model_data in data_object:
            repo_id = model_data.get("repo_id")
            key = model_data.get("key")

            if key in self._tracked_model_keys:
                continue

            calling_mechanism = model_data.get("calling_mechanism", "unknown")
            cache_hit = model_data.get("cache_hit", None)
            force_download = model_data.get("force_download", None)

            # Create section for this model
            sections = _create_model_section(
                repo_id,
                model_data,
                self._cache_scope,
                calling_mechanism=calling_mechanism,
                cache_hit=cache_hit,
                force_download=force_download,
            )
            new_components.extend(sections)
            self._tracked_model_keys.add(key)

        return new_components

    def first_time_render(self, current_card, data_object, force_refresh=False):
        """Render the card for the first time with runtime data"""
        current_card.clear()
        current_card.extend(self._header_components())

        current_card.append(Markdown("## Models Accessed at Runtime"))

        new_sections = self._make_model_sections(data_object)

        if len(new_sections) == 0:
            current_card.append(
                Markdown(
                    "_No models downloaded at runtime via `snapshot_download()` or `load()` context manager_"
                )
            )
        else:
            current_card.extend(new_sections)

        current_card.extend(self._footer_components())
        current_card.refresh(force=force_refresh)
        self._rendered = True

    def data_update(self, current_card, data_object):
        """Update the card with new runtime data"""
        new_sections = self._make_model_sections(data_object)
        if len(new_sections) == 0:
            return

        # Append new sections to the card
        current_card.extend(new_sections)
        current_card.refresh()

    def on_update(self, current_card, data_object):
        """Handle card updates"""
        if not self._rendered:
            self.first_time_render(current_card, data_object, force_refresh=False)
        else:
            self.data_update(current_card, data_object)

    def on_final(self, current_card, data_object):
        """Handle final card update"""
        self._tracked_model_keys = set()
        self._rendered = False
        self.first_time_render(current_card, data_object, force_refresh=True)


class HuggingfaceHubCollector(Thread):
    """Thread to collect Hugging Face Hub model tracking data"""

    def __init__(self, refresher: CardRefresher, interval=1):
        super().__init__()
        from metaflow import current

        self.current = current
        self._interval = interval
        self._exit_event = Event()
        self._refresher = refresher

    def collect(self):
        """Collect runtime model tracking data"""
        return self.current.huggingface_hub._runtime_tracked_models

    def final_update(self):
        """Perform final update before thread exits"""
        current_card = self.current.card[self._refresher.CARD_ID]
        data = self.collect()
        if len(data) == 0:
            return
        self._refresher.on_final(current_card, data)

    def run_update(self):
        """Perform periodic update"""
        current_card = self.current.card[self._refresher.CARD_ID]
        data = self.collect()
        if len(data) == 0:
            return
        self._refresher.on_update(current_card, data)

    def run(self):
        """Main thread loop"""
        if self._refresher.CARD_ID is None:
            raise ValueError("CARD_ID must be defined")
        current_card = self.current.card[self._refresher.CARD_ID]
        self._refresher.on_startup(current_card)
        while self._exit_event.is_set() is False:
            self.run_update()
            time.sleep(self._interval)

    def stop(self):
        """Stop the collector thread"""
        if not self._exit_event.is_set():
            self._exit_event.set()
            # We expose a `final_update` so that the card can be
            # called with a `force` update so that the new card
            # is rendered when the thread is stopped.
            self.final_update()
            self.join()
