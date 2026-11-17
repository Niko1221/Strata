"""The opt-in classifier lane must not wait behind ordinary prompt reads."""
import json
import threading
import time
import unittest
import urllib.request

from serve import test_parallel as parallel_fixtures
from serve.test_parallel import FakeEngineCase


class ClassifierLane(FakeEngineCase):
    lanes = 4

    def enable(self):
        self.engine.classifier_reserved = True
        self.svc.classifier_model = "local-classifier"

    def test_last_slot_is_reserved_for_classifier(self):
        self.start(self.lanes)
        self.enable()
        ordinary = self.lanes - 1
        self.engine.slot_busy[:ordinary] = [True] * ordinary
        self.assertIsNone(self.engine.pick_slot([1, 2, 3]))
        self.assertEqual(self.engine.pick_slot([1, 2, 3], classifier=True), ordinary)
        self.engine.slot_busy[:ordinary] = [False] * ordinary

    def test_larger_classifier_preempts_and_finishes_before_ordinary_read(self):
        self.start(self.lanes)
        self.enable()
        results, errors = {}, []

        def request(name, model, text):
            body = {"model": model, "messages": [{"role": "user", "content": text}],
                    "thinking": {"type": "disabled"}, "max_tokens": 64}
            req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    value = json.load(response)
                results[name] = (time.monotonic(), value)
            except Exception as error:
                errors.append(repr(error))

        normal = threading.Thread(target=request, args=("normal", "ordinary", "normal " * 180))
        classifier = threading.Thread(target=request, args=("classifier", "local-classifier", "classify " * 240))
        normal.start()
        time.sleep(0.08)
        classifier.start()
        normal.join(12)
        classifier.join(12)
        self.assertFalse(normal.is_alive() or classifier.is_alive(), "admission deadlocked")
        self.assertEqual(errors, [])
        self.assertLess(results["classifier"][0], results["normal"][0])
        lines = self.log.read_text().splitlines()
        self.assertTrue(any(line.startswith("YIELD ") for line in lines), lines)
        self.assertTrue(any(line.startswith(f"BGEN {self.lanes - 1} ") for line in lines), lines)
        self.assertFalse(any(self.engine.slot_busy))
        self.assertFalse(self.engine.classifier_pending)

    def test_cancelled_classifier_does_not_hold_back_normal_requests(self):
        self.start(self.lanes)
        self.enable()
        cancel = threading.Event()
        errors = []
        cancel._strata_classifier = True

        def generate():
            try:
                list(self.engine.generate_batched([1, 2], 2, {}, cancel))
            except Exception as error:
                errors.append(repr(error))

        self.engine.ctl.acquire()
        thread = threading.Thread(target=generate)
        thread.start()
        time.sleep(0.05)
        cancel.set()
        thread.join(2)
        self.engine.ctl.release()
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertFalse(self.engine.classifier_pending)


class ClassifierTwoLanes(ClassifierLane):
    lanes = 2


class ClassifierThreeLanes(ClassifierLane):
    lanes = 3

    def test_two_projects_can_finish_beside_the_classifier(self):
        self.start(self.lanes)
        self.enable()
        results, errors = {}, []

        def request(name, model, text):
            body = {"model": model, "messages": [{"role": "user", "content": text}],
                    "thinking": {"type": "disabled"}, "max_tokens": 64}
            req = urllib.request.Request(self.base + "/v1/messages", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    value = json.load(response)
                results[name] = (time.monotonic(), value)
            except Exception as error:
                errors.append(repr(error))

        workers = [threading.Thread(target=request, args=(name, "ordinary", name * 180))
                   for name in ("project-A ", "project-B ")]
        for worker in workers:
            worker.start()
        time.sleep(0.08)
        classifier = threading.Thread(target=request,
                                      args=("classifier", "local-classifier", "permission " * 240))
        classifier.start()
        for worker in [*workers, classifier]:
            worker.join(20)
        self.assertFalse(any(worker.is_alive() for worker in [*workers, classifier]))
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 3)
        self.assertLess(results["classifier"][0], min(results[name][0] for name in ("project-A ", "project-B ")))
        slots = {int(line.split()[1]) for line in self.log.read_text().splitlines() if line.startswith("BGEN ")}
        self.assertEqual(slots, {0, 1, 2})
        self.assertFalse(any(self.engine.slot_busy))
        self.assertFalse(self.engine.classifier_pending)


class ClassifierCacheIsolation(unittest.TestCase):
    def test_interleaved_projects_return_to_their_cached_prefixes(self):
        helper = parallel_fixtures.PickSlot()
        engine = helper.engine(3)
        engine.classifier_reserved = True
        history_a, history_b, policy = (helper.chat(base) for base in (1000, 20000, 40000))
        engine.slot_held = [helper.held_of(history, [31, 32], [50, 51])
                            for history in (history_a, history_b, policy)]
        engine.slot_used = [1.0, 2.0, 3.0]
        for turn in range(3):
            for owner, history in ((1, history_b), (0, history_a)):
                followup = helper.branch_of(history, [600 + turn, 61, 62 + owner])
                self.assertEqual(engine.pick_slot(followup), owner)
                engine.slot_held[owner] = followup + [50, 51]
                classifier_prompt = helper.branch_of(policy, [700 + turn, 71])
                self.assertEqual(engine.pick_slot(classifier_prompt, classifier=True), 2)
                engine.slot_held[2] = classifier_prompt + [60, 61]
                self.assertEqual(engine.pick_slot(followup + [52]), owner)
