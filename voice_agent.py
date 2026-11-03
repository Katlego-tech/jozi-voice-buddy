"""Jozi Buddy: a Strands Bidi voice agent powered by Amazon Nova Sonic.

Run it with `python voice_agent.py`, talk out loud, and press Ctrl+C to end the chat.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

import boto3
from strands.bidi.agent import BidiAgent
from strands.bidi.io import AudioIO
from strands.bidi.models import BedrockNovaSonicModel, BidiModel

MODEL_ID: Final = "amazon.nova-2-5-sonic"
DEFAULT_PROFILE: Final = "jozi-buddy"
DEFAULT_REGION: Final = "eu-north-1"
VOICE: Final = "matthew"  # the one place to change Jozi Buddy's voice

_PERSONA: Final = (
    "You are Jozi Buddy, a warm and friendly voice companion based in Johannesburg, South Africa. "
    "You chat casually with the user, help them think through small decisions, and share tips about "
    "the city when it comes up. Keep spoken replies to one or two short sentences, because this is a "
    "voice conversation, not an essay."
)
_GOODBYE: Final = "If the user says goodbye, say goodbye back warmly."

SYSTEM_PROMPT: Final = "\n\n".join([_PERSONA, _GOODBYE])


@dataclass(frozen=True)
class Settings:
    """Everything that varies between runs. Only the profile and region come from the environment."""

    profile: str
    region: str
    voice: str
    model_id: str

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        """Read `AWS_PROFILE` and `JOZI_BUDDY_REGION`; an unset or empty value means the default."""
        return cls(
            profile=env.get("AWS_PROFILE") or DEFAULT_PROFILE,
            region=env.get("JOZI_BUDDY_REGION") or DEFAULT_REGION,
            voice=VOICE,
            model_id=MODEL_ID,
        )


def build_agent(
    settings: Settings,
    session: boto3.Session,
    model_factory: Callable[..., BidiModel] = BedrockNovaSonicModel,
) -> BidiAgent:
    """Build the agent on Nova Sonic. The session carries the region, so no `region=` is passed."""
    model = model_factory(model_id=settings.model_id, boto_session=session, voice=settings.voice)
    return BidiAgent(model=model, system_prompt=SYSTEM_PROMPT)


async def converse(agent: BidiAgent, audio: AudioIO) -> None:
    """Stream the microphone to the agent and its replies to the speakers until the chat ends."""
    await agent.run(inputs=[audio.input()], outputs=[audio.output()])
