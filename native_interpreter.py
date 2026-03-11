"""Native Python interpreter for DSPy RLM — no sandbox, no WASM, just exec()."""

import io
import sys
from contextlib import redirect_stdout
from typing import Any, Callable

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput


class _SubmitCalled(Exception):
    """Raised when SUBMIT() is called to break out of exec()."""

    def __init__(self, output: dict):
        self.output = output


class NativeInterpreter:
    """Native Python interpreter using exec(). No Deno, no Pyodide, no WASM.

    WARNING: This is unsandboxed — executed code has full access to the host.
    Only use with trusted LLM-generated code (e.g., data analysis tasks).
    """

    def __init__(self, tools: dict[str, Callable] | None = None, output_fields: list[dict] | None = None):
        self.tools: dict[str, Callable] = dict(tools) if tools else {}
        self.output_fields: list[dict] | None = output_fields
        self._namespace: dict[str, Any] = {}
        self._tools_registered = False

    def _make_submit_fn(self) -> Callable:
        """Create a SUBMIT function that matches the output field signature."""
        output_fields = self.output_fields or []
        field_names = [f["name"] for f in output_fields]

        def SUBMIT(*args, **kwargs):
            result = {}
            # Positional args map to field names in order
            for i, arg in enumerate(args):
                if i < len(field_names):
                    result[field_names[i]] = arg
            # Keyword args override
            result.update(kwargs)
            raise _SubmitCalled(result)

        return SUBMIT

    def start(self) -> None:
        pass

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        # Inject variables into namespace (only on first call or if new vars)
        if variables:
            self._namespace.update(variables)

        # Register tools into namespace
        if not self._tools_registered:
            self._namespace.update(self.tools)
            self._namespace["SUBMIT"] = self._make_submit_fn()
            self._tools_registered = True

        # Capture stdout
        stdout_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf):
                exec(compile(code, "<rlm>", "exec"), self._namespace)
        except _SubmitCalled as e:
            return FinalOutput(e.output)
        except SyntaxError:
            raise
        except Exception as e:
            raise CodeInterpreterError(f"{type(e).__name__}: {e}") from e

        output = stdout_buf.getvalue()
        return output if output else None

    def shutdown(self) -> None:
        self._namespace.clear()
        self._tools_registered = False
