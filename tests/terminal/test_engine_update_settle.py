"""After a loaded model is stopped for an engine update, the runtime host that ran it ends a moment later and until then still holds the shared engine lease.
Found on a real host (Boston, the app's update with a real model): the model was stopped, the update then said "a running Strata server is serving" and failed.
The gate waits, briefly, for the lease to be free."""
import unittest
from unittest import mock

from strata_control import engine_update

MODEL = {"profile": "launch-x", "state": "ready"}
LOOSE = {"profile": "a running Strata server", "state": "serving"}


class Settle(unittest.TestCase):
    def gate(self):
        events = []
        gate = engine_update.ModelGate(ask=None, say=lambda text: None, emit=events.append, allow_stop=True, stop=lambda m: None)
        gate.consent([MODEL], "plan")
        return gate, events

    def test_the_gate_waits_until_the_lease_is_free(self):
        gate, events = self.gate()
        answers = [[MODEL], [LOOSE], [LOOSE], []]            # release's own look, then the lingering host twice, then free
        calls = []

        def serving():
            calls.append(1)
            return answers.pop(0) if answers else []

        with mock.patch.object(engine_update.time, "sleep") as sleep:
            gate.release(serving)
        self.assertEqual(len(calls), 4)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual([e["phase"] for e in events if e["event"] == "models"], ["consent", "stopping", "stopped"])
        self.assertEqual([m.profile for m in gate.stopped], ["launch-x"])

    def test_something_that_never_lets_go_is_still_reported_by_the_normal_check_after_the_wait(self):
        gate, _ = self.gate()
        clock = iter([0.0, 0.1, 5.0, 25.0, 30.0, 40.0])

        def serving(_state=[0]):
            _state[0] += 1
            return [MODEL] if _state[0] == 1 else [LOOSE]

        with mock.patch.object(engine_update.time, "sleep"), mock.patch.object(engine_update.time, "monotonic", side_effect=lambda: next(clock)):
            gate.release(serving)                              # returns; updater.wait_until_idle then refuses it, as before
        self.assertTrue(gate.stopped)

    def test_nothing_stopped_means_no_waiting(self):
        gate = engine_update.ModelGate(ask=None, say=lambda text: None, allow_stop=True, stop=lambda m: None)
        with mock.patch.object(engine_update.time, "sleep") as sleep:
            gate.release(lambda: [])
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
