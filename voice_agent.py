"""Jozi Buddy: a Strands Bidi voice agent powered by Amazon Nova Sonic.

Run it with `python voice_agent.py`, talk out loud, and press Ctrl+C to end the chat.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

import boto3
import pyaudio
from botocore.exceptions import ClientError
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


class CredentialsUnavailable(Exception):
    """The session found no AWS credentials at all."""


class AudioDeviceUnavailable(Exception):
    """There is no usable default microphone or speaker."""


class ModelUnavailable(Exception):
    """The model is not offered, or not ACTIVE, in the chosen region."""


class PyAudioLike(Protocol):
    """The slice of `pyaudio.PyAudio` the device check uses, so tests can pass a fake."""

    def get_default_input_device_info(self) -> Any: ...
    def get_default_output_device_info(self) -> Any: ...
    def terminate(self) -> None: ...


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


def make_session(settings: Settings) -> boto3.Session:
    """A session for the configured profile, pinned to the configured region (not the profile's)."""
    return boto3.Session(profile_name=settings.profile, region_name=settings.region)


def check_credentials(session: boto3.Session) -> None:
    """Fail now, not mid-connect, if the aws login session is missing or expired."""
    credentials = session.get_credentials()
    if credentials is None:
        raise CredentialsUnavailable("no credentials found")
    credentials.get_frozen_credentials()  # forces a refresh; an expired login raises LoginError here


_MODEL_NOT_OFFERED: Final = frozenset({"ValidationException", "ResourceNotFoundException"})


def check_model_available(session: boto3.Session, model_id: str) -> None:
    """Ask Bedrock's control plane whether the model is ACTIVE in the session's region.

    Needed because a stream opened in a region without the model stays silent instead of failing.
    """
    try:
        response = session.client("bedrock").get_foundation_model(modelIdentifier=model_id)
    except ClientError as error:
        code = error.response.get("Error", {}).get("Code", "")
        if code in _MODEL_NOT_OFFERED:
            raise ModelUnavailable(str(error)) from error
        if code == "AccessDeniedException":
            region = session.region_name
            print(
                f"Couldn't confirm {model_id} is available in {region} ({error}); trying anyway.",
                file=sys.stderr,
            )
            return
        raise
    status = response["modelDetails"]["modelLifecycle"]["status"]
    if status != "ACTIVE":
        raise ModelUnavailable(f"model status is {status}")


def check_audio_devices(pa_factory: Callable[[], PyAudioLike] = pyaudio.PyAudio) -> None:
    """Fail before the conversation starts if there is no default microphone or speaker."""
    pa = pa_factory()
    try:
        pa.get_default_input_device_info()
        pa.get_default_output_device_info()
    except OSError as error:
        raise AudioDeviceUnavailable(str(error)) from error
    finally:
        pa.terminate()


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
