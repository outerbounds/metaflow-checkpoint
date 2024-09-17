import json
from datetime import datetime


def is_json_serializable(value):
    """Check if value is JSON serializable."""
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False


def safe_serialize(data):
    def handle_value(val):
        """Converts non-serializable values to serializable format or a placeholder."""
        if isinstance(val, datetime):
            return val.isoformat()
        elif isinstance(val, (dict, list, tuple)):
            return safe_serialize(val)
        elif is_json_serializable(val):
            return val
        else:
            return "<value cannot be JSON serialized>"

    if isinstance(data, dict):
        if len(data) == 0:
            return {}
        return {key: handle_value(val) for key, val in data.items()}
    elif isinstance(data, list):
        return [handle_value(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(handle_value(item) for item in data)
    else:
        return handle_value(data)
