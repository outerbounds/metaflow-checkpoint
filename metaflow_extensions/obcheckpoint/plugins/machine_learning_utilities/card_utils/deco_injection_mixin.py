from metaflow.exception import MetaflowException
from collections import defaultdict


class CardDecoratorInjector:
    """
    Mixin Useful for injecting @card decorators from other first class Metaflow decorators.
    """

    _first_time_init = defaultdict(dict)

    @classmethod
    def _get_first_time_init_cached_value(cls, step_name, card_id):
        return cls._first_time_init.get(step_name, {}).get(card_id, None)

    @classmethod
    def _set_first_time_init_cached_value(cls, step_name, card_id, value):
        cls._first_time_init[step_name][card_id] = value

    def _card_deco_already_attached(self, step, card_id):
        for decorator in step.decorators:
            if decorator.name == "card":
                if decorator.attributes["id"] and card_id == decorator.attributes["id"]:
                    return True
        return False

    def _get_step(self, flow, step_name):
        for step in flow:
            if step.name == step_name:
                return step
        return None

    def _first_time_init_check(self, step_dag_node, card_id):
        """ """
        return not self._card_deco_already_attached(step_dag_node, card_id)

    def attach_card_decorator(
        self,
        flow,
        step_name,
        card_id,
        card_type,
        refresh_interval=5,
    ):
        """
        This method is called `step_init` in your StepDecorator code since
        this class is used as a Mixin
        """
        from metaflow import decorators as _decorators

        if not all([card_id, card_type]):
            raise MetaflowException(
                "`card_id` and `card_type` must be set in the `CardDecoratorInjector` Mixin"
            )

        step_dag_node = self._get_step(flow, step_name)
        if (
            self._get_first_time_init_cached_value(step_name, card_id) is None
        ):  # First check class level setting.
            if self._first_time_init_check(step_dag_node, card_id):
                self._set_first_time_init_cached_value(step_name, card_id, True)
                _decorators._attach_decorators_to_step(
                    step_dag_node,
                    [
                        "card:type=%s,id=%s,refresh_interval=%s"
                        % (card_type, card_id, str(refresh_interval))
                    ],
                )
            else:
                self._set_first_time_init_cached_value(step_name, card_id, False)
