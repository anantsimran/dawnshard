"""An OpenAI-compatible `/v1/completions` endpoint with server-sent-event streaming.

The engine runs on one background thread, stepping whenever there is work. HTTP
handler threads only add requests and read their events from a per-request queue, so
the engine is the single owner of the KV cache and the scheduler; a lock serializes
`add_request` against `step`.

```
POST /v1/completions ──► encode prompt ──► engine.add_request ──► queue[id]
engine thread: step() ──► events ──► queue[event.request_id]
handler: stream=true  → "data: {choices:[{text}]}\\n\\n" per event, then "data: [DONE]"
         stream=false → one JSON body when the request finishes
```

The standard library's threading HTTP server is enough for one engine on one GPU; the
engine step, not the HTTP layer, is the bottleneck.
"""

import json
import queue
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from training.serve.engine import Engine, Event
from training.serve.sampler import SamplingParams


class EngineLoop:
    """Owns the engine thread and routes its events to waiting handlers."""

    def __init__(self, engine: Engine) -> None:
        """Start the engine thread."""
        self.engine, self.lock = engine, threading.Lock()
        self.queues: dict[str, queue.Queue[Event]] = {}
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, prompt: list[int], params: SamplingParams) -> queue.Queue[Event]:
        """Queue a request; return the queue its events arrive on."""
        with self.lock:
            request_id = self.engine.add_request(prompt=prompt, params=params)
            self.queues[request_id] = queue.Queue()
            return self.queues[request_id]

    def _run(self) -> None:
        """Step forever; sleep briefly when idle."""
        while True:
            with self.lock:
                events = self.engine.step() if self.engine.has_work() else []
            for event in events:
                self.queues[event.request_id].put(event)  # noqa: NAR001
                if event.finished:
                    del self.queues[event.request_id]
            if not events:
                time.sleep(0.001)  # noqa: NAR001


def params_from(body: dict) -> SamplingParams:
    """Read OpenAI completion fields into `SamplingParams`."""
    stop = body.get("stop") or ()  # noqa: NAR001
    return SamplingParams(
        max_tokens=int(body.get("max_tokens", 16)),  # noqa: NAR001
        temperature=float(body.get("temperature", 1.0)),  # noqa: NAR001
        top_k=int(body.get("top_k", 0)),  # noqa: NAR001
        top_p=float(body.get("top_p", 1.0)),  # noqa: NAR001
        seed=body.get("seed"),  # noqa: NAR001
        stop=(stop,) if isinstance(stop, str) else tuple(stop),  # noqa: NAR001
        ignore_eos=bool(body.get("ignore_eos", False)),  # noqa: NAR001
    )


def make_handler(
    loop: EngineLoop, encode: Callable[[str], list[int]], model_name: str
) -> type[BaseHTTPRequestHandler]:
    """Return a request-handler class bound to `loop`."""

    class Handler(BaseHTTPRequestHandler):
        """POST /v1/completions; everything else is 404."""

        def do_POST(self) -> None:  # noqa: N802
            """Read the completion request, submit it, and stream or return the result."""
            if self.path != "/v1/completions":
                self.send_error(code=404)
                return
            body = json.loads(s=self.rfile.read(int(self.headers["Content-Length"])))  # noqa: NAR001
            events = loop.submit(prompt=encode(body["prompt"]), params=params_from(body=body))  # noqa: NAR001
            if body.get("stream"):  # noqa: NAR001
                self._stream(events=events)
            else:
                self._whole(events=events)

        def _chunk(self, text: str, finished: bool) -> dict:
            """One OpenAI completion object."""
            reason = "stop" if finished else None
            return {
                "object": "text_completion",
                "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": reason}],
            }

        def _stream(self, events: queue.Queue[Event]) -> None:
            """Write each event as a server-sent event, then `[DONE]`."""
            self.send_response(code=200)
            self.send_header(keyword="Content-Type", value="text/event-stream")
            self.end_headers()
            while True:
                event = events.get()
                chunk = json.dumps(obj=self._chunk(text=event.text, finished=event.finished))
                self.wfile.write(f"data: {chunk}\n\n".encode())  # noqa: NAR001
                self.wfile.flush()
                if event.finished:
                    break
            self.wfile.write(b"data: [DONE]\n\n")  # noqa: NAR001

        def _whole(self, events: queue.Queue[Event]) -> None:
            """Wait for the request to finish and write one JSON body."""
            text = ""
            while True:
                event = events.get()
                text += event.text
                if event.finished:
                    break
            payload = json.dumps(obj=self._chunk(text=text, finished=True)).encode()
            self.send_response(code=200)
            self.send_header(keyword="Content-Type", value="application/json")
            self.send_header(keyword="Content-Length", value=str(len(payload)))  # noqa: NAR001
            self.end_headers()
            self.wfile.write(payload)  # noqa: NAR001

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            """Silence per-request logging; the engine is the thing to watch."""

    return Handler


def serve(
    engine: Engine, encode: Callable[[str], list[int]], model_name: str, host: str, port: int
) -> ThreadingHTTPServer:
    """Start the engine thread and return a server ready for `serve_forever()`."""
    loop = EngineLoop(engine=engine)
    return ThreadingHTTPServer(  # noqa: NAR001
        (host, port), make_handler(loop=loop, encode=encode, model_name=model_name)
    )  # noqa: NAR001
