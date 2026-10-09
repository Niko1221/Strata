"""Synthetic cached-video serving and HTTP caps; no native model inference."""
import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from serve.frontend import ChatTemplate
from serve.media import MediaKind, read_bundle
from serve.server import ByteTokenizer, EngineStarting, MockEngine, Service, Vision, request_body_limit, serve
from serve.video import VIDEO_PROFILE, VideoError, VideoPolicy, VideoRequestBudget
from serve.test_video_encoder import clip_info, ready_encoder, synthetic_video


class QwenByteTokenizer(ByteTokenizer):
    SPECIALS=ByteTokenizer.SPECIALS+["<|video_pad|>"]
    mapping={259:248053,260:248056,261:248054,262:248057}
    def encode(self,*args,**kwargs):
        return [self.mapping.get(i,i) for i in super().encode(*args,**kwargs)]
    def decode(self,ids,*args,**kwargs):
        reverse={v:k for k,v in self.mapping.items()}
        return super().decode([reverse.get(i,i) for i in ids],*args,**kwargs)


def video_message(source):
    return [{"role":"user","content":[{"type":"video","source":source}]}]


class VideoPreparation(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.tok=QwenByteTokenizer();self.engine=MockEngine(self.tok,"ok")
        self.engine.info={"media_sve":2,"media_profile":VIDEO_PROFILE,"vocab":300000}
        self.svc=Service(self.engine,self.tok,ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        self.addCleanup(self.svc.drop_embeddings)
        self.bridge=ready_encoder(self.root);self.addCleanup(self.bridge.close)
        self.svc.vision=SimpleNamespace(video=self.bridge,dir=self.root)
        self.source=self.root/"clip";self.source.write_bytes(b"synthetic cache source, not a container")
        self.digest=hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.info=clip_info();self.bundle=synthetic_video(self.info,self.tok)
        budget=VideoRequestBudget(self.bridge.policy);self.info.charge(budget)
        def exchange(command,budget):
            from serve.media import write_bundle
            with (self.root/command.split()[2]).open("wb") as stream:
                write_bundle(stream,self.bundle)
            return "VOK 24 3 0"
        self.bridge.encode(self.root/"unused",self.digest,self.info,budget,exchange,self.tok)

    def test_video_only_survives_template_and_publishes_owned_bound_artifact(self):
        ids,thinking,max_new=self.svc.prepare(video_message(str(self.source)),None,{"enable_thinking":False},10)
        owned=self.svc.embeddings.owner
        self.assertTrue(owned.path.exists())
        with owned.path.open("rb") as source:
            media=read_bundle(source,qwen4=True)
        self.assertEqual(tuple(ids),media.tokens)
        self.assertEqual(len(media.spans),3)
        self.assertTrue(all(s.kind==MediaKind.VIDEO for s in media.spans))
        path=owned.path;self.svc.drop_embeddings()
        self.assertFalse(path.exists())
        self.assertEqual(self.bridge.quota.used,self.bridge.cache_used)

    def test_preparing_status_covers_media_artifact_setup(self):
        original = self.bridge.request_artifact
        observed = []
        def check(bundle, limits):
            live = self.svc.metrics()["live"]
            observed.append((live["state"], live["video_progress"]["stage"]))
            return original(bundle, limits)
        self.bridge.request_artifact = check
        try:
            self.svc.prepare(video_message(str(self.source)), None, {}, 10)
        finally:
            self.bridge.request_artifact = original
        self.assertEqual(observed, [("preparing_video", "assembling_prompt")])

    def test_run_and_abandoned_prepare_both_drop_previous_artifact(self):
        ids,thinking,max_new=self.svc.prepare(video_message(str(self.source)),None,{},10)
        path=self.svc.embeddings.owner.path
        list(self.svc.run(ids,thinking,None,max_new,{},threading.Event()))
        self.assertFalse(path.exists())
        self.svc.prepare(video_message(str(self.source)),None,{},10)
        path=self.svc.embeddings.owner.path
        self.svc.prepare([{"role":"user","content":"text only"}],None,{},10)
        self.assertFalse(path.exists())

    def test_ordered_repeated_clips_keep_every_temporal_group(self):
        parts=[{"type":"video","source":str(self.source)},
               {"type":"text","text":"then"},
               {"type":"video","source":str(self.source)}]
        ids,_,_=self.svc.prepare([{"role":"user","content":parts}],None,{},10)
        with self.svc.embeddings.owner.path.open("rb") as source:
            media=read_bundle(source,qwen4=True)
        self.assertEqual(len(media.spans),6)
        self.assertEqual(tuple(ids),media.tokens)

    def test_capability_and_readiness_fail_closed(self):
        self.assertTrue(self.svc.video_status()["available"])
        for facts in ({}, {"media_sve":2,"media_profile":"wrong"}):
            self.engine.info=facts
            self.assertFalse(self.svc.video_status()["available"])
            with self.assertRaises(EngineStarting):
                self.svc.prepare(video_message(str(self.source)),None,{},10)
        self.engine.info={"media_sve":2,"media_profile":VIDEO_PROFILE}
        self.engine.starting=True
        self.assertFalse(self.svc.video_status()["available"])

    def test_fifo_wait_can_be_cancelled_without_publishing(self):
        cancel=threading.Event()
        self.svc.fifo.acquire()
        timer=threading.Timer(.15,cancel.set);timer.start()
        try:
            with self.assertRaises(VideoError):
                self.svc.prepare(video_message(str(self.source)),None,{},10,cancel=cancel)
        finally:
            self.svc.fifo.release();timer.join()
        self.assertIsNone(getattr(self.svc.embeddings,"owner",None))
        self.assertEqual(self.bridge.quota.used,self.bridge.cache_used)


class VideoProgressProtocol(unittest.TestCase):
    def test_encoder_progress_lines_precede_final_reply(self):
        from io import StringIO

        class Process:
            def __init__(self):
                self.stdin = StringIO()
                self.stdout = StringIO("VPROG 1 3\nVPROG 2 3\nVPROG 3 3\nVOK 24 3 0\n")
            def poll(self): return None
            def kill(self): raise AssertionError("valid progress must not kill the encoder")
            def wait(self, timeout=None): return 0

        vision = object.__new__(Vision)
        vision.proc, vision.stopped = Process(), False
        progress = []
        reply = vision._video_exchange("ENCV frames output", VideoRequestBudget(VideoPolicy()),
                                       lambda done, total: progress.append((done, total)))
        self.assertEqual(reply, "VOK 24 3 0")
        self.assertEqual(progress, [(1, 3), (2, 3), (3, 3)])


class BodyCap(unittest.TestCase):
    def setUp(self):
        tok=ByteTokenizer();self.svc=Service(MockEngine(tok,"ok"),tok,ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        self.svc.start_telemetry=lambda:None
        self.http=serve(self.svc,port=0)
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.port=self.http.server_address[1]

    def post(self,body,headers=None,path="/v1/chat/completions"):
        conn=http.client.HTTPConnection("127.0.0.1",self.port,timeout=3)
        self.addCleanup(conn.close)
        conn.request("POST",path,body,headers or {"Content-Type":"application/json"})
        response=conn.getresponse();data=response.read()
        return response.status,json.loads(data)

    def test_configuration_rejects_bool_zero_fraction_and_negative(self):
        self.assertIsNone(request_body_limit(None));self.assertEqual(request_body_limit(12),12)
        for value in (True,False,0,-1,1.5,"100"):
            with self.subTest(value=value),self.assertRaises(ValueError):
                request_body_limit(value)

    def test_413_before_reading_claimed_oversized_body(self):
        self.svc.max_request_body_bytes=128
        status,error=self.post(b"",{"Content-Length":"1000000000","Content-Type":"application/json"})
        self.assertEqual(status,413)
        self.assertIn("max_request_body_bytes",error["error"]["message"])

    def test_negative_length_and_unset_legacy_body(self):
        status,_=self.post(b"",{"Content-Length":"-1"})
        self.assertEqual(status,400)
        self.svc.max_request_body_bytes=None
        body=json.dumps({"messages":[{"role":"user","content":"x"*10000}],"max_tokens":10})
        status,_=self.post(body)
        self.assertEqual(status,200)

    def test_video_count_tokens_and_unavailable_video_have_explicit_errors(self):
        messages=[{"role":"user","content":[{"type":"video_url","video_url":{"url":"https://example.com/v.mp4"}}]}]
        status,error=self.post(json.dumps({"messages":messages}),
                               path="/v1/messages/count_tokens")
        self.assertEqual(status,400)
        self.assertIn("count_tokens for video",error["error"]["message"])
        status,error=self.post(json.dumps({"messages":messages,"max_tokens":10}))
        self.assertEqual(status,400)
        self.assertIn("video is disabled",error["error"]["message"])


if __name__ == "__main__":
    unittest.main()
