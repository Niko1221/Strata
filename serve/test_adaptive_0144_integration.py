"""CPU-only ownership regressions for the local adaptive / official .44.1 composition."""
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve import server
from serve.frontend import ChatTemplate
from serve.test_vision_elastic import ElasticVision
from serve.test_vram import VramEngine
from serve import test_images as image_fixtures
from serve.test_images import fake_args, images_api

ROOT = Path(__file__).resolve().parents[1]


class Adaptive0144Integration(unittest.TestCase):
    def service(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        tok = server.ByteTokenizer()
        engine = VramEngine(tok)
        engine.loaded = True
        vision = ElasticVision(tmp.name)
        svc = server.Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'), vision=vision)
        svc.free_vram_mib = lambda: 3000
        self.addCleanup(svc.drop_embeddings)
        return svc

    def test_idle_vision_cannot_reclaim_a_tool_owners_vram(self):
        svc = self.service()
        with svc.fifo:
            svc._vision_up()
        svc.last_image_at = 0
        before = list(svc.engine.calls)
        with mock.patch.object(svc.tool_leases, 'blocked', return_value=True):
            self.assertEqual(svc._vision_idle_down(), 'busy')
        self.assertEqual(svc.vision.unloads, 0)
        self.assertEqual(svc.engine.calls, before)
        self.assertTrue(svc.vision_held)

    def test_pending_lease_allows_only_existing_image_preparation(self):
        svc = self.service()
        with svc.preparation_reservation():
            with mock.patch.object(svc.tool_leases, 'blocked', return_value=True):
                with svc._vision_turn():
                    svc._vision_up()
        self.assertEqual(svc.vision.starts, 1)
        self.assertEqual(svc.preparing_requests, 0)
        with mock.patch.object(svc.tool_leases, 'blocked', return_value=True):
            with self.assertRaises(server.ModelBusy):
                with svc._vision_turn():
                    self.fail('unreserved image reached the encoder')
        self.assertFalse(svc.fifo.locked())

    def test_tool_picture_starts_elastic_encoder_with_owned_turn(self):
        svc = self.service()
        observed = []
        encode = svc.vision.encode
        def inspect(source):
            observed.append((svc.fifo.locked(), svc.vision.alive(), svc.preparing_requests))
            return encode(source)
        svc.vision.encode = inspect
        messages = [{'role': 'tool', 'content': [{'type': 'image', 'source': 'x.png'}]}]
        svc.encode_prompt = lambda *_: [1]
        with mock.patch.object(server, 'images_of', return_value=[]):
            svc.prepare(messages, None, {})
        self.assertEqual(observed, [(True, True, 1)])
        self.assertEqual(svc.vision.starts, 1)
        self.assertEqual(messages[0]['content'][0]['type'], 'image')

    def test_cancelled_adaptive_image_is_also_a_cancelled_upstream_prepare(self):
        svc = self.service()
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(server.ClientGone):
            with svc._vision_turn(cancel):
                self.fail('cancelled turn was admitted')
        self.assertEqual(svc.vision.starts, 0)
        self.assertFalse(svc.fifo.locked())

    def test_parking_releases_elastic_hold_without_loading_encoder_on_return(self):
        svc = self.service()
        svc.vision.up = svc.vision_held = True
        svc.engine.restart = mock.Mock(side_effect=lambda **_: setattr(svc.engine, 'loaded', True))
        cancel = threading.Event()
        with svc.fifo:
            svc._parking_unload(cancel)
            self.assertFalse(svc.vision_held)
            svc._parking_reload(cancel)
        self.assertEqual(svc.vision.starts, 0)
        self.assertTrue(svc.loaded())

    def test_guarded_startup_returns_for_new_admission_before_any_retry(self):
        proc = mock.Mock(stdin=io.StringIO(), stdout=io.StringIO(), returncode=0)
        proc.poll.return_value = 0
        with mock.patch.object(server, 'popen', return_value=proc) as spawn, \
                mock.patch.object(server, 'contain'), \
                mock.patch.object(server, 'transient_start_failure', return_value=True) as transient, \
                mock.patch.object(server.StrataEngine, '_ready_pump', side_effect=lambda _, q: q.put(None)):
            with self.assertRaises(server.EngineDied):
                server.StrataEngine('unused', [], startup_cancel=threading.Event())
        spawn.assert_called_once()
        transient.assert_not_called()


class EmbeddedImageOwnership(unittest.TestCase):
    def setUp(self):
        # Reuse upstream's fake CLI and real HTTP route; no native engine or GPU.
        self.http = image_fixtures.OverHttp('test_b64')
        self.http.setUp()
        self.addCleanup(self.http.tearDown)

    def test_adaptive_modes_refuse_before_any_image_gpu_turn(self):
        svc = self.http.svc
        modes = [
            mock.patch.object(svc, 'memory_policy', object()),
            mock.patch.object(svc.tool_leases, 'enabled', True),
            mock.patch.object(svc.idle_parking, 'enabled', True),
            mock.patch.object(svc, 'coadaptive', mock.Mock(active=True)),
        ]
        for mode in modes:
            with self.subTest(mode=mode.attribute), mode, \
                    mock.patch.object(images_api, 'gpu_turn') as gpu_turn, \
                    mock.patch.object(svc.images, 'run') as image_run:
                with self.assertRaisesRegex(ValueError, 'supervisor resource broker'):
                    svc.check_embedded_image_ownership()
                code, body = self.http.post({'prompt': 'a red fox', 'size': '512x512'})
                self.assertEqual(code, 400, body)
                self.assertIn('supervisor resource broker', body['error']['message'])
                gpu_turn.assert_not_called()
                image_run.assert_not_called()
                self.assertFalse(svc.fifo.locked())

    @mock.patch.object(images_api, 'sd_cli_args', fake_args)
    def test_ordinary_image_route_remains_available(self):
        self.http.svc.check_embedded_image_ownership()
        code, body = self.http.post({'prompt': 'a red fox', 'seed': 11, 'size': '512x512'})
        self.assertEqual(code, 200, body)
        self.assertEqual(len(body['data']), 1)
        self.assertEqual(body['strata']['gpu_policy'], 'keep')
        self.assertFalse(self.http.svc.fifo.locked())


if __name__ == '__main__':
    unittest.main()
