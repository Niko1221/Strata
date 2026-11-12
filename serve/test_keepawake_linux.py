"""Linux keep-awake backend and native desktop tests.

Run the private-bus tests with::

    python -m unittest serve.test_keepawake_linux -v

The native desktop check is opt-in::

    STRATA_TEST_KEEP_AWAKE_DESKTOP=1 python -m unittest serve.test_keepawake_linux.DesktopTests -v
"""
from __future__ import annotations

import json
import os
import queue
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

try:
    from jeepney import DBusAddress, HeaderFields, MessageType, new_method_return, new_signal
    from jeepney.bus_messages import message_bus
    from jeepney.io.blocking import DBusConnection, Proxy, open_dbus_connection
except ImportError:  # Keep test discovery useful before Linux runtime requirements are installed.
    DBusAddress = DBusConnection = HeaderFields = MessageType = new_method_return = new_signal = message_bus = Proxy = None
    open_dbus_connection = None

from serve import keepawake

ROOT = Path(__file__).resolve().parents[1]
PORTAL = "org.freedesktop.portal.Desktop"
LOGIN1 = "org.freedesktop.login1"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
REQUEST_IFACE = "org.freedesktop.portal.Request"
INHIBIT_IFACE = "org.freedesktop.portal.Inhibit"
LOGIN1_PATH = "/org/freedesktop/login1"
LOGIN1_IFACE = "org.freedesktop.login1.Manager"
OWNER = '''import json, sys
from serve.keepawake import KeepAwake
ka = KeepAwake.create(True, say=lambda *a, **k: None)
print(json.dumps(ka._backend.name if ka else None), flush=True)
try:
    for command in sys.stdin:
        if command.strip() == "acquire":
            ka.acquire()
        elif command.strip() == "release":
            ka.release()
        print("done", flush=True)
finally:
    if ka:
        ka.close()
'''
PORTAL_ACTIVATION_SERVICE = '''import os, sys
log_path = sys.argv[1]
def log(value):
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(value + "\\n")
from jeepney import DBusAddress, HeaderFields, MessageType, new_method_return, new_signal
from jeepney.bus_messages import message_bus
from jeepney.io.blocking import Proxy, open_dbus_connection
os.environ["DBUS_SESSION_BUS_ADDRESS"] = os.environ["DBUS_STARTER_ADDRESS"]
connection = open_dbus_connection(bus="SESSION")
Proxy(message_bus, connection, timeout=1).RequestName("org.freedesktop.portal.Desktop", 0)
while True:
    try:
        message = connection.receive(timeout=10)
    except TimeoutError:
        continue
    if message.header.message_type != MessageType.method_call:
        continue
    fields = message.header.fields
    method = fields.get(HeaderFields.member)
    if fields.get(HeaderFields.interface) == "org.freedesktop.portal.Inhibit" and method == "Inhibit":
        _window, _flags, options = message.body
        token = options["handle_token"][1]
        sender = fields[HeaderFields.sender]
        sender_path = sender[1:].replace(".", "_")
        handle = f"/org/freedesktop/portal/desktop/request/{sender_path}/{token}"
        connection.send(new_signal(DBusAddress(handle, bus_name="org.freedesktop.portal.Desktop",
                                                interface="org.freedesktop.portal.Request"),
                                   "Response", "ua{sv}", (0, {})))
        connection.send(new_method_return(message, "o", (handle,)))
        log("inhibit")
    elif fields.get(HeaderFields.interface) == "org.freedesktop.portal.Request" and method == "Close":
        connection.send(new_method_return(message))
        log("close")
        break
connection.close()
'''


def wait_for(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@unittest.skipUnless(sys.platform == "linux" and shutil.which("dbus-daemon") and open_dbus_connection,
                     "requires Linux, Jeepney, and dbus-daemon")
class DbusBackendTests(unittest.TestCase):
    def setUp(self):
        self.bus = PrivateBus()
        self.addCleanup(lambda: self.bus.close())

    def test_connection_bounds_hello_and_runs_jeepneys_normal_initialization(self):
        opened = []
        original_receive = DBusConnection.receive

        def drop_hello(connection, message, serial=None):
            self.assertEqual(message.header.fields.get(HeaderFields.member), "Hello")
            opened.append(connection)

        def bounded_receive(connection, *, timeout=None):
            if timeout is None:
                raise AssertionError("Hello wait did not receive a timeout")
            return original_receive(connection, timeout=timeout)

        with mock.patch.object(keepawake, "DBUS_CALL_S", 0.05), \
                mock.patch.object(DBusConnection, "send_message", drop_hello), \
                mock.patch.object(DBusConnection, "receive", bounded_receive):
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                keepawake._open_dbus_connection("SESSION")
            elapsed = time.monotonic() - started

        self.assertEqual(len(opened), 1)
        self.assertTrue(hasattr(opened[0], "bus_proxy"))
        self.assertEqual(opened[0].sock.fileno(), -1)
        self.assertIsNone(opened[0].selector.get_map())
        self.assertGreaterEqual(elapsed, 0.04)
        self.assertLess(elapsed, 1.0)

    def use_bus(self, activatable=(), *, activate_portal=False):
        self.bus.close()
        self.bus = PrivateBus(activatable=activatable, activate_portal=activate_portal)

    @staticmethod
    def close_child(owner):
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=2)
        for stream in (owner.stdin, owner.stdout, owner.stderr):
            if stream is not None:
                stream.close()

    def test_portal_is_selected_before_logind_and_request_is_closed_on_release(self):
        service = self.bus.own(PORTAL, LOGIN1)
        inhibitor = keepawake._PortalInhibitor()
        self.assertTrue(inhibitor.test())
        controller = keepawake.KeepAwake(inhibitor, say=mock.Mock())
        self.addCleanup(controller.close)

        controller.acquire()
        self.assertTrue(service.wait_for("Inhibit"))
        inhibit = service.calls_named("Inhibit")[0]
        self.assertEqual(inhibit.header.fields[HeaderFields.signature], "sua{sv}")
        window, flags, options = inhibit.body
        self.assertEqual(window, "")
        self.assertEqual(flags, 4)
        self.assertEqual(options["reason"], ("s", f"Inference running or queued (pid {os.getpid()})"))
        self.assertEqual(options["handle_token"][0], "s")
        self.assertEqual(service.calls_named("Inhibit", LOGIN1_IFACE), [])

        controller.release()
        self.assertTrue(service.wait_for("Close"))
        close = service.calls_named("Close")[0]
        self.assertEqual(close.header.fields[HeaderFields.path], service.request_path)
        controller.close()

    def test_portal_ignores_response_for_another_request_path(self):
        service = self.bus.own(PORTAL)
        service.wrong_portal_response = True
        inhibitor = keepawake._PortalInhibitor()
        self.addCleanup(inhibitor.release)

        self.assertTrue(inhibitor.test())
        self.assertTrue(inhibitor.acquire(threading.Event()))
        self.assertEqual(inhibitor._handle, service.request_path)

    def test_portal_accepts_a_handle_that_does_not_follow_the_token_convention(self):
        service = self.bus.own(PORTAL)
        service.alternate_portal_path = True
        inhibitor = keepawake._PortalInhibitor()
        self.addCleanup(inhibitor.release)

        self.assertTrue(inhibitor.acquire(threading.Event()))
        self.assertEqual(inhibitor._handle, service.request_path)

    def test_portal_rejects_failed_response_and_closes_the_pending_request(self):
        service = self.bus.own(PORTAL)
        service.portal_response = 2
        inhibitor = keepawake._PortalInhibitor()
        self.addCleanup(inhibitor.release)

        self.assertTrue(inhibitor.test())
        self.assertFalse(inhibitor.acquire(threading.Event()))
        self.assertTrue(service.wait_for("Close"))
        self.assertIsNone(inhibitor._connection)

    def test_portal_response_timeout_closes_the_pending_request(self):
        service = self.bus.own(PORTAL)
        service.send_portal_response = False
        inhibitor = keepawake._PortalInhibitor()
        self.addCleanup(inhibitor.release)

        with mock.patch.object(keepawake, "PORTAL_RESPONSE_S", 0.05):
            self.assertTrue(inhibitor.test())
            self.assertFalse(inhibitor.acquire(threading.Event()))
        self.assertTrue(service.wait_for("Close"))
        self.assertIsNone(inhibitor._connection)

    def test_portal_cancellation_while_waiting_for_response_closes_request(self):
        service = self.bus.own(PORTAL)
        service.send_portal_response = False
        inhibitor = keepawake._PortalInhibitor()
        self.addCleanup(inhibitor.release)
        cancel = threading.Event()
        result = []
        worker = threading.Thread(target=lambda: result.append(inhibitor.acquire(cancel)), daemon=True)

        self.assertTrue(inhibitor.test())
        worker.start()
        self.assertTrue(service.wait_for("Inhibit"))
        cancel.set()
        worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [False])
        self.assertTrue(service.wait_for("Close"))

    def test_portal_owner_loss_is_detected_and_drops_the_session_lease(self):
        service = self.bus.own(PORTAL)
        inhibitor = keepawake._PortalInhibitor()
        self.addCleanup(inhibitor.release)
        self.assertTrue(inhibitor.test())
        self.assertTrue(inhibitor.acquire(threading.Event()))

        self.bus.release_name(PORTAL)
        self.assertTrue(wait_for(lambda: not inhibitor._healthy()))
        self.assertFalse(inhibitor.acquire(threading.Event()))
        self.assertIsNone(inhibitor._connection)
        self.assertEqual(service.calls_named("Close"), [])

    def test_logind_uses_idle_block_and_holds_a_close_on_exec_descriptor(self):
        service = self.bus.own(LOGIN1)
        inhibitor = keepawake._SystemdInhibitor()
        self.addCleanup(inhibitor.release)

        self.assertTrue(inhibitor.test())
        self.assertTrue(inhibitor.acquire(threading.Event()))
        fd = inhibitor._fd
        self.assertIsInstance(fd, int)
        call = service.calls_named("Inhibit", LOGIN1_IFACE)[0]
        self.assertEqual(call.header.fields[HeaderFields.signature], "ssss")
        self.assertEqual(call.body, ("idle", "Strata", f"Inference running or queued (pid {os.getpid()})", "block"))
        self.assertFalse(os.get_inheritable(fd))

        child = subprocess.run(
            [sys.executable, "-c", "import os,sys\ntry: os.fstat(int(sys.argv[1]))\nexcept OSError: raise SystemExit(0)\nraise SystemExit(1)", str(fd)],
            close_fds=False, check=False, timeout=2,
        )
        self.assertEqual(child.returncode, 0, "the inhibitor descriptor leaked into an exec'd child")
        self.assertTrue(inhibitor.acquire(threading.Event()))
        self.assertEqual(len(service.calls_named("Inhibit", LOGIN1_IFACE)), 1)

        inhibitor.release()
        with self.assertRaises(OSError):
            os.fstat(fd)
        self.assertTrue(service.wait_for_systemd_release())

    def test_portal_connection_closes_when_its_owner_process_crashes(self):
        service = self.bus.own(PORTAL)
        owner = subprocess.Popen(
            [sys.executable, "-u", "-c", '''import sys, threading
from serve.keepawake import _PortalInhibitor
inhibitor = _PortalInhibitor()
assert inhibitor.test() and inhibitor.acquire(threading.Event())
print(inhibitor._connection.unique_name, flush=True)
sys.stdin.buffer.read()
'''],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT,
        )
        self.addCleanup(self.close_child, owner)

        client_name = owner.stdout.readline().strip()
        self.assertTrue(client_name.startswith(":"))
        self.assertTrue(service.wait_for("Inhibit"))
        owner.kill()
        owner.wait(timeout=2)

        self.assertTrue(service.wait_for_owner_disconnect(client_name))
        self.assertEqual(service.calls_named("Close"), [])
        self.assertNotIn(client_name, service.active_portal_owners)

    def test_logind_descriptor_closes_automatically_when_its_owner_crashes(self):
        service = self.bus.own(LOGIN1)
        owner = subprocess.Popen(
            [sys.executable, "-u", "-c", '''import sys, threading
from serve.keepawake import _SystemdInhibitor
inhibitor = _SystemdInhibitor()
assert inhibitor.test() and inhibitor.acquire(threading.Event())
print("held", flush=True)
sys.stdin.buffer.read()
'''],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT,
        )
        self.addCleanup(self.close_child, owner)

        self.assertEqual(owner.stdout.readline().strip(), "held")
        self.assertTrue(service.wait_for("Inhibit", interface=LOGIN1_IFACE))
        owner.kill()
        owner.wait(timeout=2)

        self.assertTrue(service.wait_for_systemd_release())

    def test_backend_selection_uses_systemd_when_portal_is_not_owned(self):
        self.bus.own(LOGIN1)
        controller = keepawake.KeepAwake.create(True, say=mock.Mock())
        self.assertIsNotNone(controller)
        self.addCleanup(controller.close)
        self.assertEqual(controller._backend.name, "systemd idle")

    def test_portal_discovery_uses_runtime_bus_when_session_address_is_missing(self):
        self.bus.own(PORTAL)
        with mock.patch.dict(os.environ, {
            "XDG_RUNTIME_DIR": str(self.bus.runtime_dir),
        }, clear=True), mock.patch.object(keepawake.os, "getuid", return_value=0):
            self.assertTrue(keepawake._PortalInhibitor().test())

    def test_root_without_session_environment_has_no_guessed_session_bus(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(keepawake.os, "getuid", return_value=0):
            self.assertIsNone(keepawake._session_bus_address())

    def test_portal_discovery_skips_root_without_a_session_bus(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(keepawake.os, "getuid", return_value=0):
            self.assertFalse(keepawake._PortalInhibitor().test())

    def test_session_bus_address_defaults_to_the_process_runtime_bus(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(keepawake.os, "getuid", return_value=1000):
            self.assertEqual(keepawake._session_bus_address(), "unix:path=/run/user/1000/bus")

    def test_session_bus_address_escapes_runtime_path_separators(self):
        with mock.patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/run/user/1000,desktop"}, clear=True):
            self.assertEqual(keepawake._session_bus_address(), "unix:path=/run/user/1000%2Cdesktop/bus")

    def test_activatable_portal_is_selected_before_an_owned_logind_service(self):
        self.use_bus(activatable=(PORTAL,))
        service = self.bus.own(LOGIN1)
        with open_dbus_connection(bus="SESSION") as connection:
            self.assertFalse(Proxy(message_bus, connection, timeout=1).NameHasOwner(PORTAL)[0])

        controller = keepawake.KeepAwake.create(True, say=mock.Mock())
        self.assertIsNotNone(controller)
        self.addCleanup(controller.close)
        self.assertEqual(controller._backend.name, "XDG Desktop Portal suspend")
        self.assertEqual(service.calls_named("Inhibit", LOGIN1_IFACE), [])

    def test_activatable_portal_starts_and_acquires_its_request(self):
        self.use_bus(activate_portal=True)
        log_path = self.bus.activation_log
        with open_dbus_connection(bus="SESSION") as connection:
            self.assertFalse(Proxy(message_bus, connection, timeout=1).NameHasOwner(PORTAL)[0])

        controller = keepawake.KeepAwake.create(True, say=mock.Mock())
        self.assertIsNotNone(controller)
        self.addCleanup(controller.close)
        self.assertEqual(controller._backend.name, "XDG Desktop Portal suspend")
        controller.acquire()
        self.assertIsNotNone(controller._backend._handle)
        self.assertTrue(controller._backend._healthy())
        self.assertTrue(wait_for(lambda: log_path.exists() and "inhibit" in log_path.read_text()))
        controller.release()
        self.assertTrue(wait_for(lambda: log_path.exists() and "close" in log_path.read_text()))
        controller.close()

    def test_backend_selection_disables_keepawake_when_neither_service_is_owned(self):
        self.assertIsNone(keepawake.KeepAwake.create(True, say=mock.Mock()))


class PrivateBus:
    """One isolated message bus exposed as both session and system bus."""

    def __init__(self, activatable=(), *, activate_portal=False):
        self._tempdir = tempfile.TemporaryDirectory(prefix="strata-dbus-test-")
        root = Path(self._tempdir.name)
        service_dir = root / "services"
        service_dir.mkdir()
        self.runtime_dir = root / "runtime"
        self.runtime_dir.mkdir()
        self.activation_log = root / "portal-activation.log"
        if activate_portal:
            activatable = (*activatable, PORTAL)
        for name in activatable:
            if activate_portal and name == PORTAL:
                service = root / "fake-portal.py"
                service.write_text(PORTAL_ACTIVATION_SERVICE, encoding="utf-8")
                executable = f"{sys.executable} {service} {self.activation_log}"
            else:
                executable = "/bin/true"
            (service_dir / f"{name}.service").write_text(
                f"[D-BUS Service]\nName={name}\nExec={executable}\n", encoding="utf-8",
            )
        config = root / "session.conf"
        config.write_text(
            "<busconfig>"
            "<type>session</type>"
            f"<listen>unix:path={self.runtime_dir / 'bus'}</listen>"
            "<auth>EXTERNAL</auth>"
            f"<servicedir>{service_dir}</servicedir>"
            "<policy context='default'>"
            "<allow send_destination='*' eavesdrop='true'/>"
            "<allow eavesdrop='true'/>"
            "<allow own='*'/>"
            "</policy>"
            "</busconfig>",
            encoding="utf-8",
        )
        self.daemon = subprocess.Popen(
            ["dbus-daemon", f"--config-file={config}", "--nofork", "--print-address=1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.address = self.daemon.stdout.readline().strip()
        if not self.address:
            stderr = self.daemon.stderr.read()
            raise RuntimeError(f"dbus-daemon did not provide a private bus address: {stderr}")
        self.env = mock.patch.dict(os.environ, {
            "DBUS_SESSION_BUS_ADDRESS": self.address,
            "DBUS_SYSTEM_BUS_ADDRESS": self.address,
        })
        self.env.start()
        self.service = None
        self._closed = False

    def own(self, *names):
        self.service = FakeService(names)
        return self.service

    def release_name(self, name):
        if self.service is None:
            raise RuntimeError("the private bus has no service connection")
        self.service.release_name(name)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.service is not None:
            self.service.close()
        self.env.stop()
        if self.daemon.poll() is None:
            self.daemon.terminate()
            try:
                self.daemon.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.daemon.kill()
                self.daemon.wait(timeout=1)
        if self.daemon.stdout:
            self.daemon.stdout.close()
        if self.daemon.stderr:
            self.daemon.stderr.close()
        self._tempdir.cleanup()


class FakeService:
    def __init__(self, names):
        self.connection = open_dbus_connection(bus="SESSION", enable_fds=True)
        self.proxy = Proxy(message_bus, self.connection, timeout=1)
        from jeepney import MatchRule
        owner_changes = MatchRule(type="signal", sender="org.freedesktop.DBus",
                                  interface="org.freedesktop.DBus", member="NameOwnerChanged")
        self.proxy.AddMatch(owner_changes)
        for name in names:
            self.proxy.RequestName(name, 0)
        self.names = set(names)
        self.calls = []
        self._calls_lock = threading.Lock()
        self._stop = threading.Event()
        self._release_names = queue.Queue()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self.request_path = None
        self.portal_owner = None
        self.active_portal_owners = set()
        self.disconnected_portal_owners = set()
        self._systemd_release_events = []
        self.portal_response = 0
        self.send_portal_response = True
        self.wrong_portal_response = False
        self.alternate_portal_path = False
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                name, done = self._release_names.get_nowait()
            except queue.Empty:
                pass
            else:
                self.proxy.ReleaseName(name)
                done.set()
            try:
                message = self.connection.receive(timeout=0.05)
            except TimeoutError:
                continue
            except (OSError, RuntimeError, ValueError):
                return
            fields = message.header.fields
            if message.header.message_type == MessageType.signal \
                    and fields.get(HeaderFields.interface) == "org.freedesktop.DBus" \
                    and fields.get(HeaderFields.member) == "NameOwnerChanged":
                if len(message.body) == 3 and message.body[0] in self.active_portal_owners \
                        and message.body[2] == "":
                    self.disconnected_portal_owners.add(message.body[0])
                    self.active_portal_owners.discard(message.body[0])
                continue
            if message.header.message_type != MessageType.method_call:
                continue
            member = fields.get(HeaderFields.member)
            interface = fields.get(HeaderFields.interface)
            path = fields.get(HeaderFields.path)
            with self._calls_lock:
                self.calls.append(message)

            if interface == INHIBIT_IFACE and member == "Inhibit" and path == PORTAL_PATH:
                window, flags, options = message.body
                token = options["handle_token"][1]
                sender = fields[HeaderFields.sender]
                sender_path = sender[1:].replace(".", "_")
                self.request_path = (
                    "/org/freedesktop/portal/desktop/request/legacy/handle"
                    if self.alternate_portal_path
                    else f"/org/freedesktop/portal/desktop/request/{sender_path}/{token}"
                )
                self.portal_owner = sender
                self.active_portal_owners.add(sender)
                self._respond_to_portal_inhibit(message)
            elif interface == LOGIN1_IFACE and member == "Inhibit" and path == LOGIN1_PATH:
                read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
                released = threading.Event()
                self._systemd_release_events.append(released)
                try:
                    self.connection.send(new_method_return(message, "h", (write_fd,)))
                finally:
                    os.close(write_fd)
                threading.Thread(target=self._watch_systemd_release, args=(read_fd, released), daemon=True).start()
            elif interface == REQUEST_IFACE and member == "Close":
                self.active_portal_owners.discard(self.portal_owner)
                self.connection.send(new_method_return(message))
            else:
                self.connection.send(new_method_return(message))

    def _respond_to_portal_inhibit(self, message):
        if self.wrong_portal_response:
            wrong = self.request_path + "_unrelated"
            signal = new_signal(DBusAddress(wrong, bus_name=PORTAL, interface=REQUEST_IFACE),
                                "Response", "ua{sv}", (0, {}))
            self.connection.send(signal)
        self.connection.send(new_method_return(message, "o", (self.request_path,)))
        if self.send_portal_response:
            signal = new_signal(DBusAddress(self.request_path, bus_name=PORTAL, interface=REQUEST_IFACE),
                                "Response", "ua{sv}", (self.portal_response, {}))
            self.connection.send(signal)

    def calls_named(self, name, interface=None):
        with self._calls_lock:
            return [
                message for message in self.calls
                if message.header.fields.get(HeaderFields.member) == name
                and (interface is None or message.header.fields.get(HeaderFields.interface) == interface)
            ]

    def wait_for(self, name, interface=None, timeout=2.0):
        return wait_for(lambda: bool(self.calls_named(name, interface)), timeout)

    def wait_for_owner_disconnect(self, name, timeout=2.0):
        return wait_for(lambda: name in self.disconnected_portal_owners, timeout)

    @staticmethod
    def _watch_systemd_release(read_fd, released):
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(read_fd, selectors.EVENT_READ)
                if selector.select(5) and os.read(read_fd, 1) == b"":
                    released.set()
        finally:
            os.close(read_fd)

    def wait_for_systemd_release(self, timeout=2.0):
        return wait_for(lambda: any(event.is_set() for event in self._systemd_release_events), timeout)

    def release_name(self, name):
        done = threading.Event()
        self._release_names.put((name, done))
        if not done.wait(2):
            raise TimeoutError(f"fake D-Bus service did not release {name}")

    def close(self):
        self._stop.set()
        try:
            self.connection.close()
        except (OSError, RuntimeError):
            pass
        self._thread.join(timeout=1)


@unittest.skipUnless(sys.platform == "linux" and os.environ.get("STRATA_TEST_KEEP_AWAKE_DESKTOP") == "1",
                     "requires an opt-in Linux desktop inhibition check")
class DesktopTests(unittest.TestCase):
    def test_selected_linux_inhibitor_follows_requests_and_owner_death(self):
        env = {**os.environ, "LC_ALL": "C"}
        owner = subprocess.Popen([sys.executable, "-u", "-c", OWNER], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ROOT)

        def reply():
            with selectors.DefaultSelector() as selector:
                selector.register(owner.stdout, selectors.EVENT_READ)
                self.assertTrue(selector.select(10), "keep-awake owner did not respond")
                return owner.stdout.readline().strip()

        def command(value):
            owner.stdin.write(value + "\n")
            owner.stdin.flush()
            self.assertEqual(reply(), "done")

        reason = f"Inference running or queued (pid {owner.pid})"

        def holds():
            if backend == "XDG Desktop Portal suspend":
                gnome = subprocess.check_output(
                    ["gnome-session-inhibit", "--list"], env=env, text=True, timeout=3)
                own = [line for line in gnome.splitlines() if reason in line]
                for line in own:
                    self.assertIn("suspend", line.lower())
                return len(own)
            systemd = subprocess.check_output(
                ["systemd-inhibit", "--list", "--no-pager", "--no-legend"], env=env, text=True, timeout=3)
            own = [line for line in systemd.splitlines() if line.startswith("Strata ") and reason in line]
            for line in own:
                self.assertIn("idle", line.split())
                self.assertIn("block", line.split())
            return len(own)

        def wait_for(expected):
            deadline = time.monotonic() + 6
            while time.monotonic() < deadline:
                actual = holds()
                if actual == expected:
                    return
                time.sleep(0.05)
            self.fail(f"own inhibitors: expected {expected}, got {actual}")

        try:
            backend = json.loads(reply())
            self.assertIn(backend, ("XDG Desktop Portal suspend", "systemd idle"))
            if backend.startswith("XDG"):
                if "GNOME" not in os.environ.get("XDG_CURRENT_DESKTOP", "").upper():
                    self.skipTest("the portal backend is being checked through GNOME's native inhibitor list")
                if not shutil.which("gnome-session-inhibit"):
                    self.skipTest("gnome-session-inhibit is needed to inspect GNOME's native inhibitor list")
                if not keepawake._bus_has_service("SESSION", "org.gnome.SessionManager"):
                    self.skipTest("GNOME SessionManager is not available on the session bus")
            elif not shutil.which("systemd-inhibit"):
                self.skipTest("systemd-inhibit is needed to inspect logind's inhibitor list")
            wait_for(0)
            command("acquire")
            command("acquire")
            wait_for(1)
            hold_until = time.monotonic() + float(os.environ.get("STRATA_TEST_KEEP_AWAKE_HOLD_S", "0"))
            while time.monotonic() < hold_until:
                self.assertEqual(holds(), 1)
                time.sleep(min(30, max(0, hold_until - time.monotonic())))
            command("release")
            wait_for(1)                       # another active or queued request still needs the hold
            command("release")
            wait_for(0)
            command("acquire")
            wait_for(1)
            owner.kill()                       # no atexit: leases must end through EOF
            owner.wait(timeout=3)
            wait_for(0)
        finally:
            if owner.poll() is None:
                owner.stdin.close()
                try:
                    owner.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    owner.kill()
                    owner.wait(timeout=3)
            owner.stdin.close()
            owner.stdout.close()
            owner.stderr.close()


if __name__ == "__main__":
    unittest.main()
