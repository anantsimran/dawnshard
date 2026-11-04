import json
import threading
import urllib.request

from training.model.qwen3 import QWEN3_TINY
from training.model.qwen3_serve import build_engine, encode_bytes
from training.serve.api import serve
from training.serve.scheduler import SchedulerConfig


def test_streaming_completion_over_http():
    engine = build_engine(config=QWEN3_TINY, num_blocks=128, scheduler=SchedulerConfig(), prefix_caching=True)
    server = serve(engine=engine, encode=encode_bytes, model_name="tiny", host="127.0.0.1", port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1/completions"
        body = json.dumps({"prompt": "hello", "max_tokens": 5, "temperature": 0, "stream": True, "ignore_eos": True}).encode()
        request = urllib.request.Request(url=url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            lines = [line.decode().strip() for line in response if line.strip()]
        chunks = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
        assert len(chunks) == 5 and lines[-1] == "data: [DONE]"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        whole = json.dumps({"prompt": "hello", "max_tokens": 5, "temperature": 0, "ignore_eos": True}).encode()
        with urllib.request.urlopen(urllib.request.Request(url=url, data=whole), timeout=60) as response:
            text = json.loads(response.read())["choices"][0]["text"]
        assert text == "".join(c["choices"][0]["text"] for c in chunks)
    finally:
        server.shutdown()
