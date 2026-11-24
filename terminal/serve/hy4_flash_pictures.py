"""Image geometry for the native hy4-flash SigLIP2 encoder."""
import io
import math
import os
import struct
import tempfile
from pathlib import Path


class Hy4FlashPictures:
    def __init__(self, tmp_dir=None):
        self.image_token = 120120
        self.dir = Path(tmp_dir or tempfile.gettempdir())

    def span(self, width, height):
        return (height // 32) * (width // 32 + 1)

    def prepare(self, data):
        from PIL import Image
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise ValueError(f"hy4-flash could not read the picture ({error})") from error
        with im:
            w, h = im.size
            if max(w, h) > 200 * min(w, h):
                raise ValueError("hy4-flash image aspect ratio exceeds 200")
            width, height = round(w / 32) * 32, round(h / 32) * 32
            if width * height > 4194304:
                beta = math.sqrt((w * h) / 4194304)
                width, height = (max(32, math.floor(n / beta / 32) * 32) for n in (w, h))
            elif width * height < 4096:
                scale = math.sqrt(4096 / (w * h))
                width, height = (math.ceil(n * scale / 32) * 32 for n in (w, h))
            rgb = im.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            return width, height, rgb.tobytes()

    def alive(self) -> bool:
        return True

    def restart(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def encode(self, source) -> None:
        """Checks that a picture can be read and sized (the Service's pre-check of tool pictures); nothing is stored."""
        data = source if isinstance(source, bytes) else None
        if data is None:
            from serve.server import Vision
            data = Vision.load(source)
        self.prepare(data)

    def build(self, ids: list[int], pictures: list[bytes]) -> tuple[list[int], str]:
        """The prompt ids with each placeholder expanded to its picture's span, and the path of the DSI1 file for the GENI line.
        `pictures`: the bytes of every image of the request, in prompt order."""
        marks = [i for i, t in enumerate(ids) if t == self.image_token]
        if len(marks) != len(pictures):
            raise ValueError("the prompt and its images do not match")
        out: list[int] = []
        entries = []
        last = 0
        for pos, data in zip(marks, pictures):
            w, h, rgb = self.prepare(data)
            out += ids[last:pos]
            entries.append((len(out), w, h, rgb))
            out += [self.image_token] * self.span(w, h)
            last = pos + 1
        out += ids[last:]
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="strata-hy4-pics-", suffix=".dsi", dir=str(self.dir))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(b"DSI1" + struct.pack("<i", len(entries)))
                for start, w, h, rgb in entries:
                    f.write(struct.pack("<qii", start, w, h) + rgb)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return out, path
