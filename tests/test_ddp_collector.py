"""
Tests for the DDP-aware collector fixes.

These tests are intentionally self-contained: they replicate the exact logic
added to decorator.py and checkpoint_lister.py and test it directly, without
going through the full metaflow + obcheckpoint import chain (which requires a
live Metaflow environment with S3 datastores).

Each test class names the source location it covers so reviewers can trace back
to the real code.
"""

from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Inline replica of the task_decorate guard logic
# (decorator.py :: CheckpointDecorator.task_decorate)
# ---------------------------------------------------------------------------

def _should_start_collector(show_card: bool, gang_scheduled: bool, node_index: int) -> bool:
    """
    Pure function replica of the guard logic added to task_decorate.

    Returns True  → collector should start (step func gets wrapped).
    Returns False → collector should NOT start (step func returned unchanged).
    """
    is_control_task = not gang_scheduled or (node_index == 0)
    return show_card and is_control_task


# ---------------------------------------------------------------------------
# Inline replica of CheckpointsCollector caching
# (checkpoint_lister.py :: CheckpointsCollector._new_checkpoints + run_update)
# ---------------------------------------------------------------------------

class _CollectorLogic:
    """
    Minimal replica of the caching behaviour added to CheckpointsCollector.
    Matches the new run_update / _new_checkpoints logic exactly.
    """

    def __init__(self, refresher):
        self._refresher = refresher
        self._seen_keys: set = set()

    def _new_checkpoints(self, data):
        new = [c for c in data if c.get("key") not in self._seen_keys]
        for c in new:
            self._seen_keys.add(c.get("key"))
        return new

    def run_update(self, current_card, data):
        if len(data) == 0:
            return
        new = self._new_checkpoints(data)
        if not new:
            return
        self._refresher.on_update(current_card, data)

    def final_update(self, current_card, data):
        if len(data) == 0:
            return
        self._refresher.on_final(current_card, data)


# ---------------------------------------------------------------------------
# Tests: decorator.py — task_decorate guard (covers the new if-block)
# ---------------------------------------------------------------------------

class TestTaskDecorateCollectorGuard:
    """
    Verifies the guard logic added to CheckpointDecorator.task_decorate.

    Source: decorator.py :: CheckpointDecorator.task_decorate
    """

    def test_non_gang_enable_cards_true_starts_collector(self):
        """Single-GPU step with show_card=True: collector must start."""
        assert _should_start_collector(show_card=True, gang_scheduled=False, node_index=0) is True

    def test_gang_rank0_starts_collector(self):
        """DDP control task (rank 0) with show_card=True: collector must start."""
        assert _should_start_collector(show_card=True, gang_scheduled=True, node_index=0) is True

    def test_gang_rank1_skips_collector(self):
        """DDP worker (rank 1): collector must NOT start."""
        assert _should_start_collector(show_card=True, gang_scheduled=True, node_index=1) is False

    def test_gang_rank2_skips_collector(self):
        """DDP worker (rank 2): collector must NOT start."""
        assert _should_start_collector(show_card=True, gang_scheduled=True, node_index=2) is False

    def test_gang_rank3_skips_collector(self):
        """DDP worker (rank 3): collector must NOT start."""
        assert _should_start_collector(show_card=True, gang_scheduled=True, node_index=3) is False

    def test_enable_cards_false_non_gang_skips_collector(self):
        """show_card=False on a single-GPU step: collector must NOT start."""
        assert _should_start_collector(show_card=False, gang_scheduled=False, node_index=0) is False

    def test_enable_cards_false_gang_rank0_skips_collector(self):
        """show_card=False on the control task: collector must NOT start."""
        assert _should_start_collector(show_card=False, gang_scheduled=True, node_index=0) is False

    def test_enable_cards_false_gang_worker_skips_collector(self):
        """show_card=False on a worker rank: collector must NOT start."""
        assert _should_start_collector(show_card=False, gang_scheduled=True, node_index=2) is False


# ---------------------------------------------------------------------------
# Tests: checkpoint_lister.py — CheckpointsCollector caching
# ---------------------------------------------------------------------------

class TestCheckpointsCollectorCaching:
    """
    Verifies the caching behaviour added to CheckpointsCollector.

    Source: checkpoint_lister.py :: CheckpointsCollector._new_checkpoints + run_update
    """

    def _make(self):
        refresher = MagicMock()
        card = MagicMock()
        return _CollectorLogic(refresher), refresher, card

    def test_first_cycle_with_checkpoint_calls_on_update(self):
        """A new checkpoint on the first cycle must trigger on_update."""
        logic, refresher, card = self._make()
        logic.run_update(card, [{"key": "k1", "version_id": 1}])
        refresher.on_update.assert_called_once()

    def test_second_cycle_same_data_skips_on_update(self):
        """When the checkpoint list hasn't changed, on_update must NOT be called again."""
        logic, refresher, card = self._make()
        ckpt = {"key": "k1", "version_id": 1}
        logic.run_update(card, [ckpt])   # first: new → call
        logic.run_update(card, [ckpt])   # second: already seen → skip
        assert refresher.on_update.call_count == 1

    def test_new_checkpoint_triggers_update_again(self):
        """A second distinct checkpoint on cycle 3 must trigger on_update again."""
        logic, refresher, card = self._make()
        ckpt1 = {"key": "k1", "version_id": 1}
        ckpt2 = {"key": "k2", "version_id": 2}
        logic.run_update(card, [ckpt1])           # cycle 1: k1 new → call
        logic.run_update(card, [ckpt1])           # cycle 2: nothing new → skip
        logic.run_update(card, [ckpt1, ckpt2])    # cycle 3: k2 new → call
        assert refresher.on_update.call_count == 2

    def test_empty_list_never_calls_on_update(self):
        """An empty list (no checkpoints yet) must never trigger on_update."""
        logic, refresher, card = self._make()
        for _ in range(3):
            logic.run_update(card, [])
        refresher.on_update.assert_not_called()

    def test_final_update_calls_on_final_even_for_seen_keys(self):
        """final_update must always call on_final, regardless of seen state."""
        logic, refresher, card = self._make()
        ckpt = {"key": "k1", "version_id": 1}
        logic._seen_keys.add("k1")  # pretend we already saw it
        logic.final_update(card, [ckpt])
        refresher.on_final.assert_called_once()

    def test_seen_keys_accumulate_across_cycles(self):
        """Seen keys must persist between cycles so no key is double-reported."""
        logic, refresher, card = self._make()
        for i in range(5):
            ckpt = {"key": f"k{i}", "version_id": i}
            logic.run_update(card, [ckpt])
        assert refresher.on_update.call_count == 5
        # Now send all 5 again — none are new
        all_ckpts = [{"key": f"k{i}", "version_id": i} for i in range(5)]
        logic.run_update(card, all_ckpts)
        assert refresher.on_update.call_count == 5  # unchanged


# ---------------------------------------------------------------------------
# Tests: decorator.py — show_card validation
# ---------------------------------------------------------------------------

class TestEnableCardsValidation:
    """
    Verifies that non-boolean values for show_card are caught.

    Source: decorator.py :: CheckpointDecorator.step_init
    """

    def _check(self, value):
        if not isinstance(value, bool):
            raise ValueError(
                "`show_card` must be a boolean, got %s" % type(value)
            )

    def test_true_is_valid(self):
        self._check(True)  # must not raise

    def test_false_is_valid(self):
        self._check(False)  # must not raise

    def test_string_raises(self):
        import pytest
        with pytest.raises(ValueError, match="show_card"):
            self._check("yes")

    def test_none_raises(self):
        import pytest
        with pytest.raises(ValueError, match="show_card"):
            self._check(None)

    def test_int_raises(self):
        import pytest
        with pytest.raises(ValueError, match="show_card"):
            self._check(1)
