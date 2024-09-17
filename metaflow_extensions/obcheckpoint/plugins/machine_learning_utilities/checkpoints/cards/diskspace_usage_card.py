import json
from ...card_utils import (
    CardDecoratorInjector,
    CardRefresher,
    AsyncPeriodicRefresher,
    LineChart,
)

from metaflow.decorators import StepDecorator
from metaflow.cards import Markdown, Table, Artifact
from multiprocessing import Process
from metaflow.metaflow_current import current
from ...utils.disk_usage import usage_collectior
from ...utils.general import unit_convert
import tempfile


class DiskUsageCardRefresher(CardRefresher):

    CARD_ID = "disk_profiler"

    def __init__(self) -> None:
        self._rendered = False
        self._errored = False
        self._line_charts = {}

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

    def on_startup(self, current_card):
        current_card.append(
            Markdown(
                "# Disk Usage Profiler \n## %s[Attempt:%s]"
                % (current.pathspec, current.retry_count),
            )
        )
        current_card.append(
            Markdown(
                "_waiting for data to appear_",
            )
        )
        current_card.refresh()

    def on_update(self, current_card, data):
        """
        ```
        {
            "/dev/root": [
                {
                    "Filesystem": "/dev/root",
                    "Size": "582G",
                    "Used": "389G",
                    "Avail": "194G",
                    "Use%": "67%",
                    "Mounted": "/",
                    "SizeBytes": 624917741568,
                    "AvailBytes": 20803747840,
                    "Timestamp": "2024-08-13T10:05:20.764734"
                },
                {
                    "Filesystem": "/dev/root",
                    "Size": "582G",
                    "Used": "389G",
                    "Avail": "194G",
                    "Use%": "67%",
                    "Mounted": "/",
                    "SizeBytes": 624917741568,
                    "AvailBytes": 20803747840,
                    "Timestamp": "2024-08-13T10:05:25.766936"
                },
                {
                    "Filesystem": "/dev/root",
                    "Size": "582G",
                    "Used": "389G",
                    "Avail": "194G",
                    "Use%": "67%",
                    "Mounted": "/",
                    "SizeBytes": 624917741568,
                    "AvailBytes": 20803747840,
                    "Timestamp": "2024-08-13T10:05:30.769329"
                },
                {
                    "Filesystem": "/dev/root",
                    "Size": "582G",
                    "Used": "389G",
                    "Avail": "194G",
                    "Use%": "67%",
                    "Mounted": "/",
                    "SizeBytes": 624917741568,
                    "AvailBytes": 20803747840,
                    "Timestamp": "2024-08-13T10:05:35.771963"
                },
            ],
        }
        ```
        """
        if (self._errored and len(data) > 0) or len(self._line_charts) == 0:
            current_card.clear()
            self._errored = False
            self._line_charts = {}

        for fs, usage_list in data.items():
            if fs not in self._line_charts:
                self._line_charts[fs] = LineChart(
                    title=f"Disk Usage for {fs}",
                    xtitle="Timestamp",
                    ytitle="Size",
                    x_name="Timestamp",
                    y_name="Size",
                    x_axis_temporal=True,
                    width=600,
                    height=300,
                )
                current_card.append(self._line_charts[fs])

            self._line_charts[fs].update(
                [
                    {
                        "Timestamp": x["Timestamp"],
                        "Size": unit_convert(x["AvailBytes"], "B", "GB"),
                    }
                    for x in usage_list
                ]
            )
        current_card.refresh()


class DiskUsageProfilerDecorator(StepDecorator, CardDecoratorInjector):

    name = "disk_profiler"

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        self.attach_card_decorator(
            flow, step_name, DiskUsageCardRefresher.CARD_ID, "blank", refresh_interval=2
        )
        self._tempfile = None

    def task_pre_step(
        self,
        step_name,
        task_datastore,
        metadata,
        run_id,
        task_id,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
        inputs,
    ):
        self._tempfile = tempfile.NamedTemporaryFile(
            prefix="mf_disk_usage", suffix=".json"
        )
        process = Process(target=usage_collectior, args=(self._tempfile.name, 5))
        process.daemon = True
        process.start()

    def task_decorate(
        self, step_func, flow, graph, retry_count, max_user_code_retries, ubf_context
    ):
        def _wrapped_step_func(*args, **kwargs):
            async_disk_usage_refresher = AsyncPeriodicRefresher(
                DiskUsageCardRefresher(),
                updater_interval=5,
                collector_interval=5,
                file_name=self._tempfile.name,
            )

            try:
                async_disk_usage_refresher.start()
                return step_func(*args, **kwargs)
            finally:
                flow._disk_usage_stats = (
                    async_disk_usage_refresher._collector_thread.read()
                )
                async_disk_usage_refresher.stop()

        return _wrapped_step_func
