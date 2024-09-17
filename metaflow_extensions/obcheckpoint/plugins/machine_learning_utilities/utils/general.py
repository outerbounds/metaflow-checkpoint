import os
from collections import namedtuple
import json
import datetime

PathSize = namedtuple(
    "PathSize", ["path", "calculated_size", "early_stopping", "not_found"]
)
# calculated_size : size in bytes


def replace_start_and_end_slash(string):
    if string.startswith("/"):
        string = string[1:]
    if string.endswith("/"):
        string = string[:-1]
    return string


def get_path_size(start_path, early_stopping_limit=None) -> PathSize:
    total_size = 0
    if not os.path.exists(start_path):
        return PathSize(
            path=start_path, calculated_size=0, early_stopping=False, not_found=True
        )
    if os.path.isfile(start_path):
        return PathSize(
            path=start_path,
            calculated_size=os.path.getsize(start_path),
            early_stopping=False,
            not_found=False,
        )
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
                if (
                    early_stopping_limit is not None
                    and total_size >= early_stopping_limit
                ):
                    return PathSize(
                        path=start_path,
                        calculated_size=total_size,
                        early_stopping=True,
                        not_found=False,
                    )

    return PathSize(
        path=start_path,
        calculated_size=total_size,
        early_stopping=False,
        not_found=False,
    )


def warning_message(message, logger=None, ts=False, prefix="[no-prefix]"):
    msg = "%s %s" % (prefix, message)
    if logger:
        logger(msg, timestamp=ts, bad=True)


def unit_convert(number, base_unit, convert_unit):
    # base_unit : GB or MB or KB or B
    # convert_unit : GB or MB or KB or B
    # number : number of base_unit
    # return : number of convert_unit
    units = ["B", "KB", "MB", "GB"]
    if base_unit not in units or convert_unit not in units:
        raise ValueError("Invalid unit")
    base_unit_index = units.index(base_unit)
    convert_unit_index = units.index(convert_unit)
    factor = pow(1024, abs(base_unit_index - convert_unit_index))
    if base_unit_index < convert_unit_index:
        return round(number / factor, 3)
    else:
        return round(number * factor, 3)


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
        if isinstance(val, datetime.datetime):
            return val.isoformat()
        elif isinstance(val, (dict, list, tuple)):
            return safe_serialize(val)
        elif is_json_serializable(val):
            return val
        else:
            return "<value cannot be JSON serialized>"

    if isinstance(data, dict):
        return {key: handle_value(val) for key, val in data.items()}
    elif isinstance(data, list):
        return [handle_value(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(handle_value(item) for item in data)
    else:
        return handle_value(data)
