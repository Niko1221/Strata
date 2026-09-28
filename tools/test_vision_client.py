import urllib.request
import json
import time

# Create a small dynamic PNG using different bytes
import io
try:
    from PIL import Image
    im = Image.new("RGB", (16, 16), color=(int(time.time()) % 255, 128, 64))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    import base64
    png_b64 = base64.b64encode(buf.getvalue()).decode()
except Exception:
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

payload = {
    "model": "swift-1.5-iq2_xs",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this image? One short sentence."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}}
            ]
        }
    ],
    "max_tokens": 150,
    "temperature": 0.0
}

req = urllib.request.Request(
    "http://127.0.0.1:8080/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Sending unique image request to Strata server...")
t0 = time.time()
with urllib.request.urlopen(req, timeout=30) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    msg = res["choices"][0]["message"]
    print(f"Response in {time.time() - t0:.2f}s:")
    if msg.get("reasoning_content"):
        print("Thinking:", msg["reasoning_content"][:120], "...")
    print("Answer:", msg.get("content"))
