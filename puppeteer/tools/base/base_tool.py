from abc import ABC, abstractmethod
from functools import wraps
import signal
import logging
import sys
import threading

# Check if running on Windows (signal.alarm is Unix-only)
_IS_WINDOWS = sys.platform == "win32"

class Tool(ABC):
    def __init__(self, name, description, execute_function, timeout_duration=1, **kwargs):
        super().__init__()
        self.name = name
        self.description = description
        self.execute_function = execute_function
        self.timeout_duration = timeout_duration
        if not _IS_WINDOWS:
            signal.alarm(0)

    def timeout_handler(self, signum, frame):
        raise TimeoutError(f"Tool execution timed out after {self.timeout_duration} seconds")

    def with_timeout(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _IS_WINDOWS:
                # On Windows, use threading-based timeout
                result = [None]
                exception = [None]

                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=target)
                thread.daemon = True
                thread.start()
                thread.join(timeout=self.timeout_duration)

                if thread.is_alive():
                    raise TimeoutError(f"Tool execution timed out after {self.timeout_duration} seconds")
                if exception[0] is not None:
                    raise exception[0]
                return result[0]
            else:
                # On Unix, use signal.alarm
                original_handler = signal.signal(signal.SIGALRM, self.timeout_handler)
                signal.alarm(self.timeout_duration)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, original_handler)
        return wrapper

    @abstractmethod
    def execute(self, *args, **kwargs):
        # Wrap the execute_function with timeout handling
        safe_execute = self.with_timeout(self.execute_function)
        try:
            return safe_execute(*args, **kwargs)
        except TimeoutError as e:
            logging.error(f"Timeout in {self.name}: {str(e)}")
            return False, str(e)
        except Exception as e:
            return False, f"Tool execution failed: {str(e)}"
