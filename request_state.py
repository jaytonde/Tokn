
from sequence import Sequence
import asyncio
from dataclasses import dataclass

@dataclass
class RequestState:
    seq: Sequence
    done: asyncio.Event
    output: str | None = None
    error: Exception | None = None
