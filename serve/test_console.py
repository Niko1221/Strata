"""Focused diagnostics tests; no engine, model or device required."""
import gc
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import weakref

from serve import console as module
from serve.console import Console


class ConsoleTest(unittest.TestCase):
    def setUp(self):
        # All injected writers use the production _Sink.write(self, text)
        # boundary; capture still creates real owned descriptors.
        self.stdout = self.enterContext(tempfile.TemporaryFile(mode='w+b'))
        self.stderr = self.enterContext(tempfile.TemporaryFile(mode='w+b'))
        self.enterContext(mock.patch.object(sys, 'stdout', self.stdout))
        self.enterContext(mock.patch.object(sys, 'stderr', self.stderr))

    def writer(self, write=None):
        if write is not None:
            self.enterContext(mock.patch.object(module._Sink, 'write', write))
        console = Console()
        self.addCleanup(lambda: console.close(1))
        return console

    def blocked(self):
        entered, release = threading.Event(), threading.Event()
        output = []

        def write(sink, text):
            if text == 'first':
                entered.set()
                release.wait(5)
            output.append(text)

        console = self.writer(write)
        self.addCleanup(release.set)
        console.emit('first')
        self.assertTrue(entered.wait(1))
        return console, release, output

    def test_count_capacity_and_best_effort_summary(self):
        console, release, output = self.blocked()
        for i in range(128):
            self.assertTrue(console.emit(str(i)))
        for _ in range(100):
            self.assertFalse(console.emit('dropped'))
        release.set()
        self.assertTrue(console.close(1))
        self.assertEqual([x for x in output if not x.startswith('[console:')],
                         ['first'] + list(map(str, range(128))))
        self.assertEqual(output.count('[console: dropped 100 diagnostics]\n'), 1)

    def test_byte_capacity_and_utf8_truncation(self):
        console, release, output = self.blocked()
        for _ in range(16):
            self.assertTrue(console.emit('😀' * 10000))
        self.assertFalse(console.emit('x'))
        release.set()
        self.assertTrue(console.close(1))
        lines = [s for s in output if s.startswith('😀')]
        self.assertEqual(len(lines), 16)
        self.assertTrue(all(len(s.encode('utf-8')) == 4096 for s in lines))
        self.assertTrue(all(s.endswith('...\n') for s in lines))

    def test_progress_identity_replacement_and_invalidation_reclaim_capacity(self):
        console, release, output = self.blocked()
        old, new = [], []  # Equal/unhashable keys are still distinct identities.
        self.assertTrue(console.emit('x' * 4096, old))
        for _ in range(15):
            self.assertTrue(console.emit('q' * 4096))
        self.assertFalse(console.emit('full'))
        self.assertTrue(console.emit('latest', new))
        console.discard_progress(old)
        self.assertIs(console._progress[2], new)
        console.discard_progress(new)
        self.assertTrue(console.emit('final', new))
        release.set()
        self.assertTrue(console.close(1))
        self.assertEqual(output[-1], 'final')
        self.assertNotIn('latest', output)
        self.assertNotIn('x' * 4096, output)

    def test_progress_counts_toward_record_limit(self):
        console, _, _ = self.blocked()
        key = object()
        self.assertTrue(console.emit('progress', key))
        for _ in range(127):
            self.assertTrue(console.emit('normal'))
        self.assertFalse(console.emit('overflow'))
        self.assertTrue(console.emit('updated', key))
        console.discard_progress(key)
        self.assertTrue(console.emit('reclaimed'))

    def test_blocked_sink_producers_and_close_are_bounded(self):
        console, release, _ = self.blocked()
        sink = console._sinks[0][1]
        done = threading.Event()

        def produce():
            for _ in range(1000):
                console.emit('diagnostic')
            done.set()

        thread = threading.Thread(target=produce, daemon=True)
        thread.start()
        self.assertTrue(done.wait(1))
        start = time.monotonic()
        self.assertFalse(console.close(.02))
        self.assertLess(time.monotonic() - start, .5)
        self.assertFalse(sink.raw.closed)
        self.assertFalse(console.emit('after close'))
        release.set()
        self.assertTrue(console.close(1))
        thread.join(1)

    def test_failure_is_local_and_new_destination_recovers(self):
        output = []
        attempts = []

        def write(sink, text):
            attempts.append(text)
            if text == 'bad':
                raise OSError('broken destination')
            output.append(text)

        console = self.writer(write)
        console.emit('bad')
        console.emit('same destination')
        with tempfile.TemporaryFile() as new, mock.patch.object(sys, 'stdout', new):
            console.emit('new destination')
        self.assertTrue(console.close(1))
        self.assertEqual(attempts, ['bad', 'same destination', 'new destination'])
        self.assertEqual(output, ['same destination', 'new destination'])

    def test_module_lines_routing_and_progress(self):
        console = self.writer()
        with mock.patch.object(module, '_console', console):
            module.console_print('a', 2)
            module.console_print('err', stderr=True)
            module.progress('p\n', object())
            module.invalidate_progress(object())
            with self.assertRaises(ValueError):
                module.progress('bad', None)
        self.assertTrue(console.close(1))
        self.stdout.seek(0)
        self.stderr.seek(0)
        newline = os.linesep.encode()
        self.assertEqual(self.stdout.read(), b'a 2' + newline + b'p' + newline)
        self.assertEqual(self.stderr.read(), b'err' + newline)

    def test_capture_failure_never_falls_back_or_disables_channel(self):
        console, release, output = self.blocked()
        for missing in (None, io.StringIO()):
            with mock.patch.object(sys, 'stdout', missing):
                self.assertFalse(console.emit('unavailable'))
        self.assertTrue(console.emit('recovered'))
        release.set()
        self.assertTrue(console.close(1))
        self.assertEqual(output, ['first', 'recovered'])

    def test_cache_identity_uses_two_noninheritable_owners_under_saturation(self):
        console, release, _ = self.blocked()
        first = console._sinks[0][1]
        with mock.patch.object(module.os, 'dup', wraps=os.dup) as dup:
            for _ in range(127):
                console.emit('out')
            console.emit('err', stderr=True)
            self.assertFalse(console.emit('overflow'))
            self.assertEqual(dup.call_count, 1)
        second = console._sinks[1][1]
        self.assertIs(console._sinks[0][1], first)
        self.assertNotEqual(first.raw.fileno(), second.raw.fileno())
        self.assertFalse(os.get_inheritable(first.raw.fileno()))
        self.assertFalse(os.get_inheritable(second.raw.fileno()))
        release.set()
        self.assertTrue(console.close(1))

    def test_a_b_a_switch_retains_records_then_releases_owners(self):
        console, release, _ = self.blocked()
        first_released, second_released = threading.Event(), threading.Event()
        first = weakref.ref(console._sinks[0][1], lambda _: first_released.set())
        with tempfile.TemporaryFile() as other:
            with mock.patch.object(sys, 'stdout', other):
                console.emit('B')
                second = weakref.ref(console._sinks[0][1], lambda _: second_released.set())
            console.emit('A again')
            third = weakref.ref(console._sinks[0][1])
        self.assertIsNotNone(first())
        self.assertIsNotNone(second())
        self.assertIsNot(first(), third())
        release.set()
        # Superseded owners release as their records finish, without close
        # clearing the caches. The current destination remains cached.
        self.assertTrue(first_released.wait(1))
        self.assertTrue(second_released.wait(1))
        self.assertIsNone(first())
        self.assertIsNone(second())
        self.assertFalse(console._closed)
        self.assertIsNotNone(third())
        self.assertIs(console._sinks[0][1], third())
        self.assertTrue(console.close(1))
        gc.collect()
        self.assertIsNone(third())

    def test_close_and_fd_reuse_during_partial_write_keep_original_destination(self):
        entered, release = threading.Event(), threading.Event()
        original_write = module._Sink.write
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            original = open(path / 'original', 'w+b', buffering=0)
            self.addCleanup(original.close)
            fd = original.fileno()

            class PartialRaw:
                def __init__(self, raw):
                    self.raw = raw
                    self.first = True

                def close(self):
                    self.raw.close()

                def write(self, data):
                    result = self.raw.write(data[:1])
                    if self.first:
                        self.first = False
                        entered.set()
                        release.wait(5)
                    return result

            def write(sink, text):
                sink.raw = PartialRaw(sink.raw)
                original_write(sink, text)

            console = self.writer(write)
            self.addCleanup(release.set)
            with mock.patch.object(sys, 'stdout', original):
                self.assertTrue(console.emit('original destination'))
            self.assertTrue(entered.wait(1))
            original.close()
            with open(path / 'replacement', 'w+b', buffering=0) as replacement:
                self.assertEqual(replacement.fileno(), fd)
                self.assertFalse(console.close(.01))
                release.set()
                self.assertTrue(console.close(1))
                replacement.write(b'new file')
            self.assertEqual((path / 'original').read_bytes(), b'original destination')
            self.assertEqual((path / 'replacement').read_bytes(), b'new file')

    def test_pipe_newlines_match_default_text_wrapper(self):
        outputs = []
        for owned in (False, True):
            reader_fd, writer_fd = os.pipe()
            with os.fdopen(reader_fd, 'rb') as reader:
                with os.fdopen(writer_fd, 'w', encoding='utf-8') as writer:
                    if owned:
                        sink = module._Sink(writer)
                        sink.write('alpha\nbeta\n')
                        sink.raw.close()
                    else:
                        writer.write('alpha\nbeta\n')
                outputs.append(reader.read())
        self.assertEqual(outputs[0], b'alpha' + os.linesep.encode() + b'beta' + os.linesep.encode())
        self.assertEqual(outputs[1], outputs[0])

    def test_stream_encoding_and_raw_transport_avoid_buffered_methods(self):
        destination = mock.Mock(encoding='ascii')
        destination.fileno.return_value = self.stdout.fileno()
        sink = module._Sink(destination)
        sink.write('aé!')
        sink.raw.close()
        self.stdout.seek(0)
        self.assertEqual(self.stdout.read(), b'a?!')
        destination.write.assert_not_called()
        destination.flush.assert_not_called()

    def test_windows_unicode_partial_writes_and_redirected_encoding(self):
        chunks = []
        raw = mock.Mock()
        raw.write.side_effect = lambda data: chunks.append(bytes(data[:1])) or 1
        with mock.patch.object(module.os, 'name', 'nt'), \
             mock.patch.object(module, 'open', return_value=raw, create=True):
            # Real dup is closed explicitly because this test replaces its raw owner.
            with mock.patch.object(module.os, 'dup', return_value=77), \
                 mock.patch.object(module.os, 'set_inheritable') as inherit:
                sink = module._Sink(self.stdout)
                sink.write('aé中!\n')
                inherit.assert_called_once_with(77, False)
        self.assertEqual(b''.join(chunks), 'aé中!\r\n'.encode('utf-8'))
        destination = mock.Mock(encoding='utf-16-le')
        destination.fileno.return_value = self.stdout.fileno()
        with mock.patch.object(module.os, 'name', 'nt'):
            sink = module._Sink(destination)
            sink.write('é中\n')
            sink.raw.close()
        self.stdout.seek(0)
        self.assertEqual(self.stdout.read(), 'é中\r\n'.encode('utf-16-le'))

    def test_raw_write_no_progress_fails_and_capture_failure_closes_duplicate(self):
        sink = module._Sink(self.stdout)
        self.addCleanup(sink.raw.close)
        for result in (0, None, OSError('broken sink')):
            with self.subTest(result=result):
                raw = mock.Mock()
                if isinstance(result, Exception):
                    raw.write.side_effect = result
                else:
                    raw.write.return_value = result
                with mock.patch.object(sink, 'raw', raw):
                    with self.assertRaises(OSError):
                        sink.write('text')
        with mock.patch.object(module, 'open', side_effect=OSError('open failed'), create=True), \
             mock.patch.object(module.os, 'close', wraps=os.close) as close:
            with self.assertRaises(OSError):
                module._Sink(self.stdout)
            close.assert_called_once()
            with self.assertRaises(OSError):
                os.fstat(close.call_args.args[0])

    def test_failed_summary_is_not_replayed(self):
        console, release, output = self.blocked()
        original = module._Sink.write
        attempts = []

        def write(sink, text):
            if text.startswith('[console:'):
                attempts.append(text)
                raise OSError('summary failed')
            original(sink, text)

        with mock.patch.object(module._Sink, 'write', write):
            for _ in range(128):
                console.emit('queued')
            self.assertFalse(console.emit('drop'))
            release.set()
            self.assertTrue(console.close(1))
        self.assertEqual(attempts, ['[console: dropped 1 diagnostics]\n'])
        self.assertEqual(len(output), 129)

    def test_lazy_singleton_and_shutdown_no_replacement(self):
        with mock.patch.object(module, '_console', None), \
             mock.patch.object(module, '_shutdown', False):
            consoles = []
            threads = [threading.Thread(target=lambda: consoles.append(module._get_console()))
                       for _ in range(16)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(1)
            self.assertEqual(len(consoles), 16)
            self.assertTrue(all(console is consoles[0] for console in consoles))
            self.assertTrue(module.shutdown(1))
            self.assertIs(module._get_console(), consoles[0])
            self.assertFalse(consoles[0].emit('closed'))
    @unittest.skipUnless(os.name == 'nt', 'Windows console API')
    def test_real_windows_console_unicode_readback(self):
        # A fresh, hidden console isolates code pages and screen contents from
        # the user's terminal. Pipes only carry the child's assertion result.
        script = r'''
import ctypes
from ctypes import wintypes
import io
import msvcrt
import os
import sys
from serve import console

class COORD(ctypes.Structure):
    _fields_ = [('X', wintypes.SHORT), ('Y', wintypes.SHORT)]

class SCREEN_INFO(ctypes.Structure):
    _fields_ = [('size', COORD), ('position', COORD), ('attributes', wintypes.WORD),
                ('window', wintypes.SMALL_RECT), ('maximum', COORD)]

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
get_info = kernel.GetConsoleScreenBufferInfo
get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(SCREEN_INFO)]
get_info.restype = wintypes.BOOL
create = kernel.CreateConsoleScreenBuffer
create.argtypes = [wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                   wintypes.DWORD, ctypes.c_void_p]
create.restype = wintypes.HANDLE
read = kernel.ReadConsoleOutputCharacterW
read.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD,
                 COORD, ctypes.POINTER(wintypes.DWORD)]
read.restype = wintypes.BOOL
cursor = kernel.SetConsoleCursorPosition
cursor.argtypes = [wintypes.HANDLE, COORD]
cursor.restype = wintypes.BOOL
set_cp = kernel.SetConsoleOutputCP
set_cp.argtypes = [wintypes.UINT]
set_cp.restype = wintypes.BOOL
close = kernel.CloseHandle
close.argtypes = [wintypes.HANDLE]
close.restype = wintypes.BOOL

def checked(ok):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())

handle = create(0xC0000000, 3, None, 1, None)
if handle == ctypes.c_void_p(-1).value:
    raise ctypes.WinError(ctypes.get_last_error())
try:
    fd = msvcrt.open_osfhandle(handle, os.O_BINARY | os.O_RDWR)
except BaseException:
    close(handle)
    raise
# fd owns the original handle; each diagnostic sink owns a duplicate.
with os.fdopen(fd, 'w', encoding='utf-8') as stream:
    with open(fd, 'wb', buffering=0, closefd=False) as raw:
        assert not isinstance(raw, io.FileIO), type(raw)
    expected = 'caf\u00e9 \u4e2d\u6587 \u03a9 \u0416'
    for codepage in (437, 936):
        checked(set_cp(codepage))
        checked(cursor(handle, COORD(0, 0)))
        writer = console.Console()
        sys.stdout = stream
        assert writer.emit(expected)
        assert writer.close(2), 'console did not drain'
        # CJK glyphs occupy two screen cells; read beyond the text and strip
        # only the untouched trailing spaces in this fresh screen buffer.
        cells = len(expected) * 2
        actual = ctypes.create_unicode_buffer(cells + 1)
        count = wintypes.DWORD()
        checked(read(handle, actual, cells, COORD(0, 0), ctypes.byref(count)))
        assert count.value > 0, count.value
        assert actual.value.rstrip(' ') == expected, (codepage, ascii(actual.value), ascii(expected))
        layouts = []
        for start_row, use_transport in ((3, False), (7, True)):
            checked(cursor(handle, COORD(0, start_row)))
            if use_transport:
                writer = console.Console()
                assert writer.emit('alpha\nbeta')
                assert writer.close(2), 'multiline output did not drain'
            else:
                stream.write('alpha\nbeta')
                stream.flush()
            info = SCREEN_INFO()
            checked(get_info(handle, ctypes.byref(info)))
            lines = []
            for row in (start_row, start_row + 1):
                line = ctypes.create_unicode_buffer(17)
                checked(read(handle, line, 16, COORD(0, row), ctypes.byref(count)))
                lines.append(line.value.rstrip(' '))
            layouts.append((lines, info.position.X, info.position.Y - start_row))
        expected_layout = (['alpha', 'beta'], 4, 1)
        assert layouts == [expected_layout, expected_layout], (codepage, layouts)
    assert console.shutdown(1)
sys.stdout = sys.__stdout__
print('console Unicode readback passed: CP437 and CP936; multiline matches TextIOWrapper, cursor=(4,1)')
'''
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        env = os.environ.copy()
        env.pop('PYTHONLEGACYWINDOWSSTDIO', None)
        result = subprocess.run(
            [sys.executable, '-B', '-c', script],
            cwd=Path(__file__).resolve().parents[1], capture_output=True,
            timeout=15, creationflags=subprocess.CREATE_NEW_CONSOLE,
            startupinfo=startup, env=env)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        self.assertIn(b'console Unicode readback passed: CP437 and CP936', result.stdout)

    @unittest.skipUnless(os.name == 'posix', 'POSIX pipe saturation')
    def test_real_full_pipe_subprocess_exits(self):
        script = r'''
import os, sys, threading
from serve import console
r, w = os.pipe()
os.set_blocking(w, False)
try:
    while True:
        os.write(w, b'x' * 4096)
except BlockingIOError:
    pass
os.set_blocking(w, True)
os.dup2(w, 1)
os.close(w)
entered = threading.Event()
original = console._Sink.write
def write(sink, text):
    entered.set()
    return original(sink, text)
console._Sink.write = write
console.console_print('blocked diagnostic')
assert entered.wait(1)
assert not console.shutdown(.02)
# Keep the pipe read end open and unread through interpreter finalization.
'''
        result = subprocess.run([sys.executable, '-c', script],
                                cwd=Path(__file__).resolve().parents[1],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
