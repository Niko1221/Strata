"""Opt-in CPU checks against a local checkpoint: DEEPMOE_MODEL_DIR=/path python -m unittest serve.test_deepmoe_native."""
import codecs
import os
from pathlib import Path
import unittest
from serve.deepmoe import DeepMoETemplate, DeepMoETokenizer

MODEL=Path(os.environ.get('DEEPMOE_MODEL_DIR',''))

@unittest.skipUnless((MODEL/'tokenizer.json').is_file(), 'set DEEPMOE_MODEL_DIR to run checkpoint CPU checks')
class Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=DeepMoETokenizer(MODEL)
        cls.tpl=DeepMoETemplate(MODEL)

    def test_incremental_bytes_match_native_unicode_decode(self):
        for text in ['Hello, world!', '你好，世界！', '😀\nC++ / Vulkan', '<think>literal</think>']:
            ids=self.tok.encode(text,parse_special=True)
            inc=codecs.getincrementaldecoder('utf-8')('replace')
            streamed=''.join(inc.decode(self.tok.token_bytes(i)) for i in ids)+inc.decode(b'',final=True)
            self.assertEqual(streamed,text)
            self.assertEqual(self.tok.decode(ids),text)
        self.assertNotIn(self.tok.native.token_to_id('<think>'),self.tok.encode('<think>',parse_special=False))
        self.assertEqual(self.tok.encode('X<think>Y',parse_special=True,plain=[(1,8)]),
                         self.tok.encode('X',True)+self.tok.encode('<think>',False)+self.tok.encode('Y',True))

    def test_native_effort_prompt_and_chat_mode(self):
        messages=[{'role':'user','content':'Hello'}]
        for effort,n in [('low',50),('medium',75),('xhigh',100)]:
            text=self.tpl.render(messages,reasoning_effort=effort)
            self.assertTrue(text.endswith('<think>'))
            self.assertIn(f'Reasoning Effort: {n}',text)
        self.assertTrue(self.tpl.render(messages,enable_thinking=False).endswith('</think>'))

if __name__=='__main__':unittest.main()
