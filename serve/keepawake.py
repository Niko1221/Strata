"""Opt-in system sleep inhibition while requests run or queue.

One worker owns the inhibitor. Requests wait for acquisition attempts, and
the last request releases them. Windows uses a thread-local execution state;
Linux selects the XDG Desktop Portal or systemd logind at startup, only when
enabled. Requests share the controller and its selected backend. Display
blanking and locking are left alone.
"""
from __future__ import annotations

import atexit
from contextlib import ExitStack, suppress
import os
import secrets
import sys
import threading
import time
from typing import Protocol
from urllib.parse import quote

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
RETRY_S = 5.0
DBUS_CALL_S = 2.0
PORTAL_RESPONSE_S = 2.0
DBUS_POLL_S = 0.1

_DBUS_NAME = "org.freedesktop.DBus"
_DBUS_PATH = "/org/freedesktop/DBus"
_PORTAL_NAME = "org.freedesktop.portal.Desktop"
_PORTAL_PATH = "/org/freedesktop/portal/desktop"
_PORTAL_REQUEST_PREFIX = _PORTAL_PATH + "/request"
_PORTAL_INHIBIT_IFACE = "org.freedesktop.portal.Inhibit"
_PORTAL_REQUEST_IFACE = "org.freedesktop.portal.Request"
_LOGIN1_NAME = "org.freedesktop.login1"
_LOGIN1_PATH = "/org/freedesktop/login1"
_LOGIN1_IFACE = "org.freedesktop.login1.Manager"
_INHIBIT_REASON = f"Inference running or queued (pid {os.getpid()})"


def _session_bus_address() -> str | None:
    address = os.environ.get("DBUS_SESSION_BUS_ADDRESS")
    if address:
        return address
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime_dir:
        uid = os.getuid()
        if uid == 0:
            # A root service cannot infer which logged-in user's bus to use.
            return None
        runtime_dir = f"/run/user/{uid}"
    return f"unix:path={quote(runtime_dir + '/bus', safe='/')}"


def _open_dbus_connection(bus: str, *, enable_fds: bool = False):
    """Open a Jeepney connection with bounded Hello and close-on-exec FD reads."""
    import socket

    from jeepney import FileDescriptor
    from jeepney.bus import get_bus
    from jeepney.fds import fds_buf_size
    from jeepney.io.blocking import DBusConnection, prep_socket, unwrap_read

    class _BoundedConnection(DBusConnection):
        def __init__(self, sock, enable_fds=False):
            try:
                super().__init__(sock, enable_fds)
            except BaseException:
                with suppress(Exception):
                    self.close()
                raise

        def send_and_get_reply(self, message, *, timeout=None):
            # Jeepney's constructor sends Hello through this method without a timeout.
            if timeout is None:
                timeout = DBUS_CALL_S
            return super().send_and_get_reply(message, timeout=timeout)

        def _read_with_fds(self):
            nbytes = self.parser.bytes_desired()
            flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)
            data, ancdata, msg_flags, _ = self.sock.recvmsg(nbytes, fds_buf_size(), flags)
            if msg_flags & getattr(socket, "MSG_CTRUNC", 0):
                self.close()
                raise RuntimeError("Unable to receive all D-Bus file descriptors")
            return unwrap_read(data), FileDescriptor.from_ancdata(ancdata)

    address = _session_bus_address() if bus == "SESSION" else bus
    if address is None:
        raise RuntimeError("No session D-Bus address is available for this process")
    sock = prep_socket(get_bus(address), enable_fds=enable_fds, timeout=DBUS_CALL_S)
    try:
        return _BoundedConnection(sock, enable_fds)
    except BaseException:
        with suppress(OSError):
            sock.close()
        raise


def _dbus_call(connection, destination, path, interface, method, signature=None, body=()):
    from jeepney import DBusAddress, new_method_call
    from jeepney.wrappers import unwrap_msg

    address = DBusAddress(path, bus_name=destination, interface=interface)
    message = new_method_call(address, method, signature, body)
    reply = connection.send_and_get_reply(message, timeout=DBUS_CALL_S)
    return unwrap_msg(reply)


def _bus_has_service(bus: str, name: str) -> bool:
    """Check whether a service is owned or D-Bus-activatable during startup selection."""
    if not sys.platform.startswith("linux"):
        return False
    connection = None
    try:
        connection = _open_dbus_connection(bus)
        owned, = _dbus_call(connection, _DBUS_NAME, _DBUS_PATH, _DBUS_NAME, "NameHasOwner", "s", (name,))
        if owned is True:
            return True
        activatable, = _dbus_call(connection, _DBUS_NAME, _DBUS_PATH, _DBUS_NAME, "ListActivatableNames")
        return name in activatable
    except Exception:  # noqa: BLE001  no bus or no Jeepney means this backend is unavailable
        return False
    finally:
        if connection is not None:
            with suppress(Exception):
                connection.close()


class _Inhibitor(Protocol):
    name: str

    def test(self) -> bool: ...

    def acquire(self, cancel: threading.Event) -> bool: ...

    def release(self) -> None: ...


class _WindowsInhibitor:
    name = "Windows system sleep"

    def __init__(self, call=None):
        self._call = call
        self._held = False

    @staticmethod
    def _windows_call():
        import ctypes
        fn = ctypes.windll.kernel32.SetThreadExecutionState
        fn.argtypes = [ctypes.c_uint32]
        fn.restype = ctypes.c_uint32
        return fn

    def test(self) -> bool:
        if not sys.platform.startswith("win"):
            return False
        try:
            if self._call is None:
                self._call = self._windows_call()
            return True
        except (ImportError, AttributeError, OSError):
            return False

    def acquire(self, cancel):
        if cancel.is_set():
            return False
        if not self._held:
            self._held = bool(self._call(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
        return self._held

    def release(self):
        if self._held:
            if not self._call(ES_CONTINUOUS):
                raise OSError("could not clear the Windows execution state")
            self._held = False


class _PortalInhibitor:
    """Keep one XDG Desktop Portal request alive on its owning session bus."""

    name = "XDG Desktop Portal suspend"

    def __init__(self):
        self._connection = None
        self._filters = None
        self._responses = None
        self._owner_changes = None
        self._owner = None
        self._handle = None
        self._owner_lost = False

    def test(self) -> bool:
        return _bus_has_service("SESSION", _PORTAL_NAME)

    def _connect(self):
        from jeepney import MatchRule
        from jeepney.wrappers import DBusErrorResponse

        connection = _open_dbus_connection("SESSION")
        filters = ExitStack()
        response_rule = MatchRule(type="signal", interface=_PORTAL_REQUEST_IFACE, member="Response",
                                  path_namespace=_PORTAL_REQUEST_PREFIX)
        owner_rule = MatchRule(type="signal", sender=_DBUS_NAME, interface=_DBUS_NAME, member="NameOwnerChanged")
        owner_rule.add_arg_condition(0, _PORTAL_NAME)
        try:
            # More than one app can receive portal responses on the session
            # bus while our method call is in flight. Keep a bounded queue and
            # correlate by the returned request path below.
            responses = filters.enter_context(connection.filter(response_rule, bufsize=256))
            owner_changes = filters.enter_context(connection.filter(owner_rule))
            _dbus_call(connection, _DBUS_NAME, _DBUS_PATH, _DBUS_NAME, "AddMatch", "s", (response_rule.serialise(),))
            _dbus_call(connection, _DBUS_NAME, _DBUS_PATH, _DBUS_NAME, "AddMatch", "s", (owner_rule.serialise(),))
            try:
                owner, = _dbus_call(
                    connection, _DBUS_NAME, _DBUS_PATH, _DBUS_NAME,
                    "GetNameOwner", "s", (_PORTAL_NAME,),
                )
            except DBusErrorResponse as exc:
                if exc.name != "org.freedesktop.DBus.Error.NameHasNoOwner":
                    raise
                # Start an activatable portal before sending Inhibit. This
                # avoids timing out while the desktop service is loading.
                _dbus_call(
                    connection, _DBUS_NAME, _DBUS_PATH, _DBUS_NAME,
                    "StartServiceByName", "su", (_PORTAL_NAME, 0),
                )
                owner, = _dbus_call(
                    connection, _DBUS_NAME, _DBUS_PATH, _DBUS_NAME,
                    "GetNameOwner", "s", (_PORTAL_NAME,),
                )
        except BaseException:
            with suppress(Exception):
                filters.close()
            with suppress(Exception):
                connection.close()
            raise
        self._connection = connection
        self._filters = filters
        self._responses = responses
        self._owner_changes = owner_changes
        self._owner = owner
        self._owner_lost = False

    def _check_owner_changes(self) -> bool:
        while self._owner_changes:
            message = self._owner_changes.popleft()
            if self._owner is not None and len(message.body) == 3 \
                    and message.body[0] == _PORTAL_NAME and message.body[2] != self._owner:
                self._owner_lost = True
        return not self._owner_lost

    def _healthy(self) -> bool:
        connection = self._connection
        if connection is None or self._handle is None:
            return False
        try:
            while True:
                connection.recv_messages(timeout=0)
        except TimeoutError:
            pass
        except Exception:  # noqa: BLE001  EOF or bus errors mean the portal lease is gone
            return False
        return self._check_owner_changes()

    def acquire(self, cancel: threading.Event) -> bool:
        from jeepney import HeaderFields

        if cancel.is_set():
            return False
        if self._handle is not None:
            if self._healthy():
                return True
            self.release()
        if cancel.is_set():
            return False

        acquired = False
        token = "strata_" + secrets.token_hex(16)
        try:
            self._connect()
            options = {
                "handle_token": ("s", token),
                "reason": ("s", _INHIBIT_REASON),
            }
            handle, = _dbus_call(
                self._connection, _PORTAL_NAME, _PORTAL_PATH, _PORTAL_INHIBIT_IFACE,
                "Inhibit", "sua{sv}", ("", 4, options),
            )
            if not isinstance(handle, str) or not handle.startswith(_PORTAL_REQUEST_PREFIX + "/"):
                return False
            self._handle = handle
            deadline = time.monotonic() + PORTAL_RESPONSE_S
            while not cancel.is_set():
                if not self._check_owner_changes():
                    return False
                while self._responses:
                    message = self._responses.popleft()
                    fields = message.header.fields
                    if fields.get(HeaderFields.path) != self._handle \
                            or fields.get(HeaderFields.sender) != self._owner:
                        continue
                    response, _results = message.body
                    if response == 0 and self._healthy():
                        acquired = True
                        return True
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                try:
                    self._connection.recv_messages(timeout=min(DBUS_POLL_S, remaining))
                except TimeoutError:
                    pass
            return False
        except Exception:  # noqa: BLE001  a failed portal request is not an inference error
            return False
        finally:
            if not acquired:
                self.release()

    def release(self) -> None:
        connection, self._connection = self._connection, None
        filters, self._filters = self._filters, None
        handle, self._handle = self._handle, None
        owner_lost, self._owner_lost = self._owner_lost, False
        self._responses = self._owner_changes = None
        self._owner = None
        if connection is None:
            return
        try:
            if handle is not None and not owner_lost:
                with suppress(Exception):
                    _dbus_call(connection, _PORTAL_NAME, handle, _PORTAL_REQUEST_IFACE, "Close")
        finally:
            try:
                if filters is not None:
                    filters.close()
            finally:
                connection.close()


class _SystemdInhibitor:
    """Hold the descriptor returned by logind's Manager.Inhibit method."""

    name = "systemd idle"

    def __init__(self):
        self._fd = None

    def test(self) -> bool:
        return _bus_has_service("SYSTEM", _LOGIN1_NAME)

    def acquire(self, cancel: threading.Event) -> bool:
        if cancel.is_set():
            return False
        if self._fd is not None:
            try:
                os.fstat(self._fd)
                return True
            except OSError:
                self._fd = None

        connection = None
        descriptor = None
        raw_fd = None
        try:
            connection = _open_dbus_connection("SYSTEM", enable_fds=True)
            (descriptor,) = _dbus_call(
                connection,
                _LOGIN1_NAME,
                _LOGIN1_PATH,
                _LOGIN1_IFACE,
                "Inhibit",
                "ssss",
                ("idle", "Strata", _INHIBIT_REASON, "block"),
            )
            raw_fd = descriptor.to_raw_fd()
            descriptor = None
            os.set_inheritable(raw_fd, False)
            if cancel.is_set():
                return False
            self._fd, raw_fd = raw_fd, None
            return True
        finally:
            if descriptor is not None:
                descriptor.close()
            if raw_fd is not None:
                with suppress(OSError):
                    os.close(raw_fd)
            if connection is not None:
                with suppress(Exception):
                    connection.close()

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is not None:
            os.close(fd)


class KeepAwake:
    _backends = (_WindowsInhibitor, _PortalInhibitor, _SystemdInhibitor)

    def __init__(self, backend: _Inhibitor, say=print):
        self._backend = backend
        self._say = say
        self._count = 0
        self._cond = threading.Condition()
        self._closed = False
        self._thread = None
        self._epoch = self._ack = 0
        self._cancel = threading.Event()
        self._failed = False
        self._release_failed = False

    @staticmethod
    def _log(say, text):
        try:
            say(f'[strata] prevent_sleep: {text}', flush=True)
        except Exception:                 # noqa: BLE001  a closed log pipe must not break inhibition
            pass

    @classmethod
    def _select_backend(cls, say):
        for backend_type in cls._backends:
            backend = backend_type()
            if backend.test():
                return backend
        cls._log(say, "no available inhibitor on this platform")
        return None

    @classmethod
    def create(cls, enabled: bool, say=print):
        """Select a backend and create the server's controller at startup."""
        if not enabled:
            return None
        backend = cls._select_backend(say)
        if backend is None:
            return None
        controller = cls(backend, say)
        atexit.register(controller.close)
        cls._log(say, 'enabled while requests run or queue: ' + backend.name)
        return controller

    def _release_backend(self):
        try:
            self._backend.release()
            self._release_failed = False
            return True
        except Exception:             # noqa: BLE001  retry cleanup without failing a request
            if not self._release_failed:
                self._release_failed = True
                self._log(self._say, f'could not release {self._backend.name}')
            return False

    def _loop(self):
        active = False
        retry_at = 0.0
        release_at = 0.0
        try:
            while True:
                with self._cond:
                    while not self._closed:
                        want = self._count > 0
                        if not want and active and time.monotonic() >= release_at:
                            self._ack = 0       # invalidate BEFORE dropping the lock for OFF
                            break
                        if want and (self._ack != self._epoch or time.monotonic() >= retry_at):
                            break
                        due = retry_at if want else release_at
                        self._cond.wait(max(0, due - time.monotonic()) if want or active else None)
                    if self._closed:
                        return
                    epoch, cancel = self._epoch, self._cancel
                if not want:
                    active = not self._release_backend()
                    release_at = time.monotonic() + RETRY_S
                    continue
                try:
                    ok = not cancel.is_set() and self._backend.acquire(cancel)
                except Exception:     # noqa: BLE001  never fail a request over this
                    ok = False
                if ok:
                    self._failed = False
                elif not cancel.is_set() and not self._failed:
                    self._failed = True
                    self._log(self._say, f'{self._backend.name} unavailable; continuing without this hold')
                active = True
                release_at = 0.0
                retry_at = time.monotonic() + RETRY_S
                with self._cond:
                    if epoch == self._epoch and not cancel.is_set():
                        self._ack = epoch        # ON attempts finished, including failures
                    self._cond.notify_all()
        finally:
            self._release_backend()
            with self._cond:
                self._closed = True
                self._cancel.set()
                self._cond.notify_all()

    def _start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="keep-awake")
            self._thread.start()

    def acquire(self):
        with self._cond:
            if self._closed:
                return
            if self._count == 0:
                self._epoch += 1
                self._cancel = threading.Event()
            self._count += 1
            epoch, cancel = self._epoch, self._cancel
            try:
                self._start()
            except RuntimeError:      # thread resources unavailable: continue unprotected
                self._closed = True
                self._thread = None
                cancel.set()
            self._cond.notify_all()
            self._cond.wait_for(lambda: self._closed or cancel.is_set() or self._ack == epoch)

    def release(self):
        with self._cond:
            self._count = max(0, self._count - 1)
            if self._count == 0:
                self._cancel.set()
            self._cond.notify_all()

    def close(self):
        with self._cond:
            self._closed = True
            self._cancel.set()
            self._cond.notify_all()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=3)
