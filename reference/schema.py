
from pydantic import List, Dict

class CompletionResponse:
    text : str
    token_generated : int
    prompt_tokens : int
    finish_reason : str  # stop/lenght


class CompletionResponse:
    pass


class OpenAICompletionRequest:
    model : str
    messages : List[Dict]
    max_tokens : int | None = None
    temperature : float = 1.0
    stream : bool = False

