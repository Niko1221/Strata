"""Real diagnostic writers with bounded completion for integration tests."""
import contextlib
import tempfile
from unittest import mock
from serve import console


@contextlib.contextmanager
def capture_console(stream):
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as destination:
        writer = console.Console()
        with contextlib.redirect_stdout(destination), mock.patch.object(console, "_console", writer):
            try:
                yield stream
            finally:
                if not writer.close(2):
                    raise AssertionError("diagnostic writer did not finish captured output")
                destination.flush()
                destination.seek(0)
                stream.write(destination.read())


@contextlib.contextmanager
def blocked_console(writer, write):
    """Inject blocking only at the same sink write boundary used in production."""
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as destination, \
            contextlib.redirect_stdout(destination), \
            mock.patch.object(console._Sink, "write", autospec=True,
                              side_effect=lambda sink, text: write(text)), \
            mock.patch.object(console, "_console", writer):
        yield
