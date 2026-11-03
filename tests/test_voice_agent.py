"""Unit tests for voice_agent. No network, microphone or AWS account is needed."""

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import boto3
import pytest
from aws_sdk_bedrock_runtime.models import AccessDeniedException, ValidationException
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    LoginRefreshRequired,
    NoCredentialsError,
    ProfileNotFound,
)
from strands.bidi.agent import BidiAgent
from strands.bidi.models import BedrockNovaSonicModel

from voice_agent import (
    DEFAULT_PROFILE,
    DEFAULT_REGION,
    MODEL_ID,
    VOICE,
    AudioDeviceUnavailable,
    CredentialsUnavailable,
    ModelUnavailable,
    Settings,
    build_agent,
    check_audio_devices,
    check_credentials,
    check_model_available,
    converse,
    find_known_error,
    main,
    make_session,
)

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


# -------------------------------------------------------------------------- preflight ----


@pytest.fixture
def isolated_aws_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point boto3 at a throwaway config with one profile, never at the real ~/.aws files."""
    config = tmp_path / "config"
    config.write_text("[profile tester]\nregion = eu-west-1\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    for var in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(var, raising=False)
    return config


def settings_for(profile: str = "tester", region: str = "eu-north-1") -> Settings:
    return Settings(profile=profile, region=region, voice=VOICE, model_id=MODEL_ID)


class TestMakeSession:
    def test_uses_the_profile_and_the_settings_region_over_the_profiles(
        self, isolated_aws_config: Path
    ) -> None:
        session = make_session(settings_for(profile="tester", region="eu-north-1"))
        assert session.profile_name == "tester"
        assert session.region_name == "eu-north-1"

    def test_unknown_profile_raises_profile_not_found(self, isolated_aws_config: Path) -> None:
        with pytest.raises(ProfileNotFound):
            make_session(settings_for(profile="no-such-profile"))


class FakeCredentials:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.frozen_calls = 0

    def get_frozen_credentials(self) -> object:
        self.frozen_calls += 1
        if self.error is not None:
            raise self.error
        return object()


class FakeBedrockClient:
    def __init__(self, status: str = "ACTIVE", error: Exception | None = None) -> None:
        self.status, self.error = status, error
        self.requested: list[str] = []

    def get_foundation_model(self, *, modelIdentifier: str) -> dict[str, Any]:  # noqa: N803 - boto3's name
        self.requested.append(modelIdentifier)
        if self.error is not None:
            raise self.error
        return {"modelDetails": {"modelId": modelIdentifier, "modelLifecycle": {"status": self.status}}}


class FakeSession:
    region_name = "eu-north-1"

    def __init__(self, credentials: FakeCredentials | None = None, bedrock: FakeBedrockClient | None = None):
        self.credentials, self.bedrock = credentials, bedrock
        self.clients: list[str] = []

    def get_credentials(self) -> FakeCredentials | None:
        return self.credentials

    def client(self, name: str) -> FakeBedrockClient:
        self.clients.append(name)
        assert self.bedrock is not None
        return self.bedrock


def client_error(code: str, message: str = "nope") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "GetFoundationModel")


class TestCheckCredentials:
    def test_no_credentials_raises_credentials_unavailable(self) -> None:
        session: Any = FakeSession(credentials=None)
        with pytest.raises(CredentialsUnavailable):
            check_credentials(session)

    def test_forces_a_refresh_now(self) -> None:
        creds = FakeCredentials()
        session: Any = FakeSession(credentials=creds)
        check_credentials(session)
        assert creds.frozen_calls == 1

    def test_expired_login_session_propagates(self) -> None:
        session: Any = FakeSession(credentials=FakeCredentials(error=LoginRefreshRequired()))
        with pytest.raises(LoginRefreshRequired):
            check_credentials(session)


class TestCheckModelAvailable:
    def test_active_model_passes_and_asks_the_bedrock_control_plane(self) -> None:
        bedrock = FakeBedrockClient(status="ACTIVE")
        session: Any = FakeSession(bedrock=bedrock)
        check_model_available(session, MODEL_ID)
        assert session.clients == ["bedrock"]
        assert bedrock.requested == [MODEL_ID]

    def test_model_that_is_not_active_is_unavailable(self) -> None:
        session: Any = FakeSession(bedrock=FakeBedrockClient(status="LEGACY"))
        with pytest.raises(ModelUnavailable):
            check_model_available(session, MODEL_ID)

    @pytest.mark.parametrize("code", ["ValidationException", "ResourceNotFoundException"])
    def test_model_not_offered_in_region_is_unavailable(self, code: str) -> None:
        session: Any = FakeSession(bedrock=FakeBedrockClient(error=client_error(code)))
        with pytest.raises(ModelUnavailable):
            check_model_available(session, MODEL_ID)

    def test_missing_permission_warns_and_carries_on(self, capsys: pytest.CaptureFixture[str]) -> None:
        error = client_error("AccessDeniedException", "not allowed")
        session: Any = FakeSession(bedrock=FakeBedrockClient(error=error))
        check_model_available(session, MODEL_ID)
        err = capsys.readouterr().err
        assert err == f"Couldn't confirm {MODEL_ID} is available in eu-north-1 ({error}); trying anyway.\n"

    def test_other_client_errors_propagate(self) -> None:
        session: Any = FakeSession(bedrock=FakeBedrockClient(error=client_error("ThrottlingException")))
        with pytest.raises(ClientError):
            check_model_available(session, MODEL_ID)

    def test_network_failure_propagates(self) -> None:
        error = EndpointConnectionError(endpoint_url="https://bedrock.eu-north-1.amazonaws.com")
        session: Any = FakeSession(bedrock=FakeBedrockClient(error=error))
        with pytest.raises(EndpointConnectionError):
            check_model_available(session, MODEL_ID)


class FakePyAudio:
    def __init__(self, fail_input: bool = False, fail_output: bool = False) -> None:
        self.fail_input, self.fail_output = fail_input, fail_output
        self.terminated = 0

    def get_default_input_device_info(self) -> dict[str, Any]:
        if self.fail_input:
            raise OSError("No Default Input Device Available")
        return {"name": "default"}

    def get_default_output_device_info(self) -> dict[str, Any]:
        if self.fail_output:
            raise OSError("No Default Output Device Available")
        return {"name": "default"}

    def terminate(self) -> None:
        self.terminated += 1


class TestCheckAudioDevices:
    def test_both_devices_present_passes_and_releases_pyaudio(self) -> None:
        pa = FakePyAudio()
        check_audio_devices(lambda: pa)
        assert pa.terminated == 1

    @pytest.mark.parametrize("which", ["input", "output"])
    def test_missing_device_is_unavailable_and_still_releases_pyaudio(self, which: str) -> None:
        pa = FakePyAudio(fail_input=which == "input", fail_output=which == "output")
        with pytest.raises(AudioDeviceUnavailable, match=f"No Default {which.title()} Device Available"):
            check_audio_devices(lambda: pa)
        assert pa.terminated == 1


# -------------------------------------------------------------- find_known_error ----


def chained(outer: BaseException, inner: BaseException) -> BaseException:
    """Build `outer` with `inner` as its __context__, the way Strands' cleanup masks errors."""
    try:
        try:
            raise inner
        except BaseException:
            raise outer  # noqa: B904 - an implicit __context__ is exactly what we are reproducing
    except BaseException as error:
        return error


class TestFindKnownError:
    def test_finds_the_bedrock_error_strands_masked(self) -> None:
        real = ValidationException(message="The provided model identifier is invalid.")
        masked = chained(RuntimeError("exceptions=[{'func_name': 'stop', ...}]"), real)
        assert find_known_error(masked) is real

    def test_an_unmasked_known_error_is_found_at_link_zero(self) -> None:
        real = AccessDeniedException(message="denied")
        assert find_known_error(real) is real

    def test_follows_explicit_cause_too(self) -> None:
        real = LoginRefreshRequired()
        outer = RuntimeError("wrapped")
        outer.__cause__ = real
        assert find_known_error(outer) is real

    def test_unrelated_chain_returns_none(self) -> None:
        assert find_known_error(chained(RuntimeError("outer"), KeyError("inner"))) is None

    def test_a_self_referencing_chain_terminates(self) -> None:
        a, b = RuntimeError("a"), RuntimeError("b")
        a.__context__, b.__context__ = b, a
        assert find_known_error(a) is None


# -------------------------------------------------------------------------- converse ----


class FakeAgent:
    def __init__(self) -> None:
        self.runs: list[tuple[list[Any], list[Any]]] = []

    async def run(self, inputs: list[Any], outputs: list[Any]) -> None:
        self.runs.append((inputs, outputs))


class FakeAudio:
    def input(self) -> str:
        return "mic"

    def output(self) -> str:
        return "speakers"


def test_converse_runs_the_agent_with_audio_in_and_out() -> None:
    agent, audio = FakeAgent(), FakeAudio()
    asyncio.run(converse(agent, audio))  # type: ignore[arg-type]
    assert agent.runs == [(["mic"], ["speakers"])]


# ------------------------------------------------------------------------------ main ----

READY = (
    "Jozi Buddy is listening (AWS profile 'jozi-buddy', eu-north-1). "
    "Start talking \u2014 press Ctrl+C to end the chat.\n"
)
GOODBYE = "\nConversation ended. Goodbye!\n"
SIGN_IN = "Can't sign in to AWS with profile 'jozi-buddy': {reason}\nRun: aws login --profile jozi-buddy\n"


class Steps:
    """Stand-ins for every step main() takes, each able to raise on demand."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.raises: dict[str, BaseException] = {}
        self.calls: list[str] = []
        self.audio_kwargs: dict[str, Any] = {}
        sentinel_session, sentinel_agent = object(), object()

        def step(name: str, result: Any = None) -> Any:
            def run(*args: Any, **kwargs: Any) -> Any:
                self.calls.append(name)
                if name in self.raises:
                    raise self.raises[name]
                return result

            return run

        def audio_io(**kwargs: Any) -> object:
            self.calls.append("AudioIO")
            self.audio_kwargs = kwargs
            return object()

        async def fake_converse(agent: Any, audio: Any) -> None:
            self.calls.append("converse")
            if "converse" in self.raises:
                raise self.raises["converse"]

        monkeypatch.setattr("voice_agent.make_session", step("make_session", sentinel_session))
        monkeypatch.setattr("voice_agent.check_credentials", step("check_credentials"))
        monkeypatch.setattr("voice_agent.check_model_available", step("check_model_available"))
        monkeypatch.setattr("voice_agent.check_audio_devices", step("check_audio_devices"))
        monkeypatch.setattr("voice_agent.build_agent", step("build_agent", sentinel_agent))
        monkeypatch.setattr("voice_agent.AudioIO", audio_io)
        monkeypatch.setattr("voice_agent.converse", fake_converse)


@pytest.fixture
def steps(monkeypatch: pytest.MonkeyPatch) -> Steps:
    return Steps(monkeypatch)


class TestMainHappyPath:
    def test_runs_every_step_in_order_and_says_goodbye(
        self, steps: Steps, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main({}) == 0
        assert steps.calls == [
            "make_session",
            "check_credentials",
            "check_model_available",
            "check_audio_devices",
            "build_agent",
            "AudioIO",
            "converse",
        ]
        assert steps.audio_kwargs == {"audio_processor": True}
        out = capsys.readouterr()
        assert out.out == READY + GOODBYE
        assert out.err == ""

    def test_ready_line_names_an_overridden_profile_and_region(
        self, steps: Steps, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main({"AWS_PROFILE": "other", "JOZI_BUDDY_REGION": "us-east-1"})
        assert "(AWS profile 'other', us-east-1)" in capsys.readouterr().out

    def test_ctrl_c_during_the_conversation_says_goodbye_and_exits_zero(
        self, steps: Steps, capsys: pytest.CaptureFixture[str]
    ) -> None:
        steps.raises["converse"] = KeyboardInterrupt()
        assert main({}) == 0
        assert capsys.readouterr().out == READY + GOODBYE

    def test_ctrl_c_during_preflight_says_goodbye_and_exits_zero(
        self, steps: Steps, capsys: pytest.CaptureFixture[str]
    ) -> None:
        steps.raises["check_audio_devices"] = KeyboardInterrupt()
        assert main({}) == 0
        assert capsys.readouterr().out == GOODBYE


class TestMainPreflightFailures:
    @pytest.mark.parametrize(
        ("step", "error", "reason"),
        [
            (
                "make_session",
                ProfileNotFound(profile="jozi-buddy"),
                "The config profile (jozi-buddy) could not be found",
            ),
            ("check_credentials", CredentialsUnavailable("no credentials found"), "no credentials found"),
            ("check_credentials", NoCredentialsError(), "Unable to locate credentials"),
            (
                "check_credentials",
                LoginRefreshRequired(),
                "Your session has expired or credentials have changed. "
                "Please reauthenticate using 'aws login'.",
            ),
        ],
    )
    def test_credential_problems_exit_2_with_the_sign_in_message(
        self, steps: Steps, capsys: pytest.CaptureFixture[str], step: str, error: Exception, reason: str
    ) -> None:
        steps.raises[step] = error
        assert main({}) == 2
        out = capsys.readouterr()
        assert out.err == SIGN_IN.format(reason=reason)
        assert out.out == ""

    def test_model_not_in_region_exits_4(self, steps: Steps, capsys: pytest.CaptureFixture[str]) -> None:
        steps.raises["check_model_available"] = ModelUnavailable("not offered")
        assert main({"JOZI_BUDDY_REGION": "eu-west-1"}) == 4
        assert capsys.readouterr().err == (
            "Amazon Nova Sonic (amazon.nova-2-5-sonic) isn't available in eu-west-1.\n"
            "Set JOZI_BUDDY_REGION to a region that offers it, such as eu-north-1.\n"
        )

    def test_no_network_exits_5(self, steps: Steps, capsys: pytest.CaptureFixture[str]) -> None:
        error = EndpointConnectionError(endpoint_url="https://bedrock.eu-north-1.amazonaws.com")
        steps.raises["check_model_available"] = error
        assert main({}) == 5
        assert capsys.readouterr().err == (
            f"Can't reach AWS in eu-north-1: {error}\n"
            "Check your internet connection, then run Jozi Buddy again.\n"
        )

    def test_no_audio_device_exits_3(self, steps: Steps, capsys: pytest.CaptureFixture[str]) -> None:
        steps.raises["check_audio_devices"] = AudioDeviceUnavailable("No Default Input Device Available")
        assert main({}) == 3
        assert capsys.readouterr().err == (
            "Can't use the microphone or speakers: No Default Input Device Available\n"
            "Check your sound settings, then run Jozi Buddy again.\n"
        )


class TestMainConversationFailures:
    def test_masked_access_denied_exits_4(self, steps: Steps, capsys: pytest.CaptureFixture[str]) -> None:
        steps.raises["converse"] = chained(
            RuntimeError("cleanup failed"), AccessDeniedException(message="no access")
        )
        assert main({}) == 4
        assert capsys.readouterr().err == (
            "Amazon Bedrock refused access to amazon.nova-2-5-sonic in eu-north-1: no access\n"
            "Check that this account can use the model in that region.\n"
        )

    def test_masked_other_bedrock_error_exits_4(
        self, steps: Steps, capsys: pytest.CaptureFixture[str]
    ) -> None:
        real = ValidationException(message="The provided model identifier is invalid.")
        steps.raises["converse"] = chained(RuntimeError("cleanup failed"), real)
        assert main({}) == 4
        assert capsys.readouterr().err == (
            "Amazon Bedrock error (ValidationException) for amazon.nova-2-5-sonic in eu-north-1: "
            "The provided model identifier is invalid.\n"
        )

    def test_session_expiring_at_a_reconnect_exits_2(
        self, steps: Steps, capsys: pytest.CaptureFixture[str]
    ) -> None:
        steps.raises["converse"] = chained(RuntimeError("cleanup failed"), LoginRefreshRequired())
        assert main({}) == 2
        reason = (
            "Your session has expired or credentials have changed. Please reauthenticate using 'aws login'."
        )
        assert capsys.readouterr().err == SIGN_IN.format(reason=reason)

    def test_an_unexpected_error_is_not_swallowed(self, steps: Steps) -> None:
        boom = RuntimeError("a real bug")
        steps.raises["converse"] = boom
        with pytest.raises(RuntimeError) as caught:
            main({})
        assert caught.value is boom


# ------------------------------------------------------------------- grounding (US2) ----

DESIGNED_PROMPT = (
    "You are Jozi Buddy, a warm and friendly voice companion based in Johannesburg, South Africa. "
    "You chat casually with the user, help them think through small decisions, and share tips about the "
    "city when it comes up. Keep spoken replies to one or two short sentences, because this is a voice "
    "conversation, not an essay.\n"
    "\n"
    "You have no internet access and no live information. Never state opening hours, prices, addresses, "
    "directions, event dates, or safety conditions as facts. If you are asked for any of these, say you "
    "can't check that right now and suggest confirming with the venue or a current local source. If you "
    "are not sure something is true, say so instead of guessing.\n"
    "\n"
    "If the user says goodbye, say goodbye back warmly."
)


class TestSystemPromptGrounding:
    def test_the_built_agent_uses_the_designed_prompt_verbatim(self, session: boto3.Session) -> None:
        agent = build_agent(Settings.from_env({}), session)
        assert agent.system_prompt == DESIGNED_PROMPT

    def test_the_prompt_states_it_has_no_live_information(self, session: boto3.Session) -> None:
        agent = build_agent(Settings.from_env({}), session)
        assert agent.system_prompt is not None
        assert "no live information" in agent.system_prompt
