

__all__ = ["current_case_kwargs", "get_current_case"]


def get_current_case(case_id=None, **location):


    from .case_context import get_current_case as _get_current_case

    return _get_current_case(case_id, **location)


def current_case_kwargs(case_id=None, **location):


    from .case_context import current_case_kwargs as _current_case_kwargs

    return _current_case_kwargs(case_id, **location)
