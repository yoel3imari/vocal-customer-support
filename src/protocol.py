"""Wire protocol shared by the browser console and the server.

Transport
---------
One WebSocket per call at ``/ws``.

* **Text frames** = JSON control messages (everything below).
* **Binary frames** = raw little-endian PCM16 mono audio, no WAV headers.
  Browser -> server: mic audio, always 16 kHz (``INPUT_SAMPLE_RATE``).
  Server -> browser: agent speech at ``sampleRate`` announced in ``state``
  (Piper voice rate, commonly 22050).

Client -> server
----------------
``{"type": "start"}``                     open a call (server sends greeting)
``{"type": "end"}``                       hang up; server replies ``call_end``
``{"type": "mute", "muted": bool}``       mic muted/unmuted
``{"type": "interrupt"}``                 caller barged in (client VAD) -> flush TTS
``{"type": "ping", "t": <n>}``            keepalive; echoed back as ``pong``

Server -> client
----------------
``{"type": "state", "state": ..., ...}``  idle|listening|thinking|speaking + metadata
``{"type": "transcript", "role": ..., "text": ..., "final": bool}``
``{"type": "metrics", "stage": ..., "ms": ...}``
``{"type": "interrupt"}``                 stop/flush queued playback now
``{"type": "error", "message": ...}``
``{"type": "call_end", "callId": ...}``
"""

from __future__ import annotations

from typing import Any, Literal

State = Literal["idle", "listening", "thinking", "speaking"]
Role = Literal["caller", "agent"]


def state(state: str, *, call_id: str | None = None, tenant: str | None = None,
          tenant_name: str | None = None, sample_rate: int | None = None,
          voice: str | None = None, detail: str | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"type": "state", "state": state}
    if call_id is not None:
        msg["callId"] = call_id
    if tenant is not None:
        msg["tenant"] = tenant
    if tenant_name is not None:
        msg["tenantName"] = tenant_name
    if sample_rate is not None:
        msg["sampleRate"] = sample_rate
    if voice is not None:
        msg["voice"] = voice
    if detail is not None:
        msg["detail"] = detail
    return msg


def transcript(role: Role, text: str, *, final: bool = True,
               interrupted: bool = False) -> dict[str, Any]:
    return {"type": "transcript", "role": role, "text": text,
            "final": final, "interrupted": interrupted}


def metrics(stage: str, ms: float, **extra: Any) -> dict[str, Any]:
    return {"type": "metrics", "stage": stage, "ms": round(ms, 1), **extra}


def interrupt() -> dict[str, Any]:
    return {"type": "interrupt"}


def error(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


def call_end(call_id: str) -> dict[str, Any]:
    return {"type": "call_end", "callId": call_id}
