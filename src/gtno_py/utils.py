import inspect


def get_context(instance: object) -> str:
    """
    Returns the class name and method name of the caller context.

    Parameters:
        instance: The instance of the class (typically `self`).

    Returns:
        str: The class name and method name in the format `ClassName.MethodName`.
    """
    frame = inspect.currentframe().f_back  # Access the caller's frame
    if frame is None:
        raise RuntimeError("Unable to determine the calling context.")

    class_name = instance.__class__.__name__
    method_name = frame.f_code.co_name
    return f"{class_name}.{method_name}"
