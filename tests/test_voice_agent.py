"""Unit tests for voice_agent. No network, microphone or AWS account is needed."""

import dataclasses
from typing import Any

import boto3
import pytest
from strands.bidi.agent import BidiAgent
from strands.bidi.models import BedrockNovaSonicModel

from voice_agent import DEFAULT_PROFILE, DEFAULT_REGION, MODEL_ID, VOICE, Settings, build_agent

# ---------------------------------------------------------------------------- settings ----


class TestSettingsFromEnv:
    def test_constants_have_the_designed_values(self) -> None:
        assert MODEL_ID == "amazon.nova-2-5-sonic"
        assert DEFAULT_PROFILE == "jozi-buddy"
        assert DEFAULT_REGION == "eu-north-1"
        assert VOICE == "matthew"

    def test_defaults_when_nothing_is_set(self) -> None:
        assert Settings.from_env({}) == Settings(
            profile="jozi-buddy", region="eu-north-1", voice="matthew", model_id="amazon.nova-2-5-sonic"
        )

    def test_aws_profile_overrides_the_profile(self) -> None:
        assert Settings.from_env({"AWS_PROFILE": "someone-else"}).profile == "someone-else"

    def test_jozi_buddy_region_overrides_the_region(self) -> None:
        assert Settings.from_env({"JOZI_BUDDY_REGION": "us-east-1"}).region == "us-east-1"

    def test_empty_values_count_as_unset(self) -> None:
        settings = Settings.from_env({"AWS_PROFILE": "", "JOZI_BUDDY_REGION": ""})
        assert (settings.profile, settings.region) == ("jozi-buddy", "eu-north-1")

    def test_standard_aws_region_variables_are_ignored(self) -> None:
        env = {"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "eu-west-1"}
        assert Settings.from_env(env).region == "eu-north-1"

    def test_voice_and_model_are_not_configurable_from_the_environment(self) -> None:
        settings = Settings.from_env({"VOICE": "tiffany", "MODEL_ID": "something-else"})
        assert (settings.voice, settings.model_id) == ("matthew", "amazon.nova-2-5-sonic")

    def test_settings_are_frozen(self) -> None:
        settings = Settings.from_env({})
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.region = "us-east-1"  # type: ignore[misc]


# ----------------------------------------------------------------------- agent wiring ----


@pytest.fixture
def session() -> boto3.Session:
    """A real boto3 session with dummy static credentials; nothing here makes a network call."""
    return boto3.Session(
        aws_access_key_id="testing", aws_secret_access_key="testing", region_name="eu-north-1"
    )


class RecordingFactory:
    """Records every call, then builds the real Nova Sonic model so the agent gets a real BidiModel."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.built: list[BedrockNovaSonicModel] = []

    def __call__(self, **kwargs: Any) -> BedrockNovaSonicModel:
        self.calls.append(kwargs)
        model = BedrockNovaSonicModel(**kwargs)
        self.built.append(model)
        return model


class TestBuildAgent:
    def test_model_factory_gets_exactly_model_id_session_and_voice(self, session: boto3.Session) -> None:
        factory = RecordingFactory()
        build_agent(Settings.from_env({}), session, model_factory=factory)
        assert factory.calls == [
            {"model_id": "amazon.nova-2-5-sonic", "boto_session": session, "voice": "matthew"}
        ]

    def test_no_region_argument_is_passed(self, session: boto3.Session) -> None:
        factory = RecordingFactory()
        build_agent(Settings.from_env({}), session, model_factory=factory)
        assert "region" not in factory.calls[0]

    def test_returns_a_bidi_agent_using_the_built_model(self, session: boto3.Session) -> None:
        factory = RecordingFactory()
        agent = build_agent(Settings.from_env({}), session, model_factory=factory)
        assert isinstance(agent, BidiAgent)
        assert agent.model is factory.built[0]

    def test_default_factory_builds_nova_sonic_in_the_session_region(self, session: boto3.Session) -> None:
        agent = build_agent(Settings.from_env({}), session)
        assert isinstance(agent.model, BedrockNovaSonicModel)
        assert agent.model.region == "eu-north-1"
