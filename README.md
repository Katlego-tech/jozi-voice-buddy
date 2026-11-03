# Jozi Buddy

A voice AI agent that runs on your laptop: talk out loud and it talks back, in near real time.

**Project page:** https://katlego-tech.github.io/jozi-voice-buddy/

## What it does

Jozi Buddy is a friendly Johannesburg-based voice companion. You chat with it through your
microphone and it answers through your speakers, in a sentence or two at a time. You can cut in
while it's talking and it stops to listen. It has no internet access, so when you ask for things it
can't know, like today's opening hours or prices, it says so instead of making something up.

## How it works

`voice_agent.py` wires three pieces together:

- **[Strands Agents](https://strandsagents.com/) Bidi API**: a `BidiAgent` streams audio both ways
  at once, which is what makes interrupting work.
- **Amazon Nova 2.5 Sonic on Amazon Bedrock**: a speech-to-speech model. Your voice goes in and a
  spoken reply comes out, with no separate transcription or text-to-speech step.
- **Strands `AudioIO`** with echo cancellation on, so it doesn't hear itself through laptop speakers.

Before it starts listening, it checks your AWS sign-in, that the model is available in your region,
and that you have a microphone and speakers, so problems show up as one clear message instead of
silence.

## Prerequisites

- **Python 3.12**
- **A microphone and speakers** (headphones are nicer, but laptop speakers work)
- **An AWS account** that can use Amazon Nova 2.5 Sonic. It's offered in `eu-north-1` and
  `us-east-1`. Usage is billed by Amazon Bedrock; see
  [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) for current Nova Sonic rates.
- **AWS CLI v2** with `aws login` (browser sign-in, no access keys)
- **PortAudio headers** on Linux (see setup), because PyAudio builds from source there

## Setup

Commands are for Linux and macOS.

1. **Get the code**

   ```bash
   git clone https://github.com/Katlego-tech/jozi-voice-buddy.git
   cd jozi-voice-buddy
   ```

2. **Install PortAudio** (Linux; on macOS use `brew install portaudio`)

   ```bash
   sudo apt install portaudio19-dev
   ```

3. **Create a virtual environment and install**

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   No `python3.12` command? If you use [uv](https://docs.astral.sh/uv/), create the environment with
   `uv venv --python 3.12 .venv` instead of the first line, and install with
   `uv pip install -r requirements.txt`.

4. **Sign in to AWS** with short-lived credentials through your browser

   ```bash
   aws configure set region eu-north-1 --profile jozi-buddy
   aws login --profile jozi-buddy
   ```

   The session lasts 12 hours and can be renewed for up to 90 days without signing in again
   (per the [Agent Toolkit for AWS setup guide](https://github.com/aws/agent-toolkit-for-aws/blob/main/setup-instructions/setup.md)).

## Run it

```bash
python voice_agent.py
```

When you see `Jozi Buddy is listening`, say hi. A `Speak…` prompt shows while it listens.

**To end the chat, press Ctrl+C.** You'll see `Conversation ended. Goodbye!`.

## Make it your own

- **Personality:** edit `_PERSONA` in `voice_agent.py`. Keep replies short; it's a voice chat.
  Leave `_GROUNDING` in place unless you want it to guess at facts.
- **Voice:** change `VOICE` in `voice_agent.py` (default `matthew`). Nova Sonic's voices include
  `tiffany`, `amy`, `olivia`, `kiara` and `arjun`; see the
  [Nova Sonic voice list](https://docs.aws.amazon.com/nova/latest/nova2-userguide/sonic-language-support.html).
- **Another AWS profile or region:** no code change needed.

  ```bash
  AWS_PROFILE=my-profile JOZI_BUDDY_REGION=us-east-1 python voice_agent.py
  ```

  `JOZI_BUDDY_REGION` is deliberately separate from `AWS_REGION`, which is often set to a region
  that doesn't offer Nova Sonic.

## Troubleshooting

| You see | What to do |
| --- | --- |
| `pip install` fails building **PyAudio** (`portaudio.h: No such file or directory`) | Install the PortAudio headers: `sudo apt install portaudio19-dev`, then rerun `pip install -r requirements.txt`. |
| About a dozen lines starting **`ALSA lib …`** when it starts (Linux) | Expected and harmless: PyAudio probing sound devices. It still works; ignore them. |
| `Can't sign in to AWS with profile '…'` | Your session expired or the profile doesn't exist. Run the `aws login --profile …` command it prints. |
| `Amazon Nova Sonic (…) isn't available in …` | That region doesn't offer the model. Use `JOZI_BUDDY_REGION=eu-north-1` (or `us-east-1`). |
| `Amazon Bedrock refused access to …` | Your account or role can't use the model in that region. Check model access in the Bedrock console and your IAM permissions. |
| `Amazon Bedrock error (…) for …` | Bedrock rejected the request for another reason, named in the brackets (for example `ThrottlingException`). Wait a moment and try again; if it repeats, the message after the colon says why. |
| `Couldn't confirm … is available …; trying anyway.` | Your credentials can't call `bedrock:GetFoundationModel`. It carries on; if it then stays silent, check the region. |
| `Can't use the microphone or speakers` | No default input or output device. Check your sound settings and that the mic isn't muted. |
| It cuts itself off a few words into each reply | Echo isn't being cancelled. Use headphones, or lower the speaker volume. |

Exit codes: `0` normal end, `1` unexpected error (Python prints a traceback; please open an issue),
`2` sign-in, `3` audio device, `4` Bedrock or region, `5` network.

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check . && ruff format --check .
pyright
```

The tests use fakes for AWS, the model and audio, so they need no network, microphone or account.
Dependencies are pinned in `requirements.txt` and `requirements-dev.txt`, compiled from the `.in`
files with `uv pip compile`.

## Credits

Built from the Mzansi AWS Community Day tutorial on voice agents with Kiro, Strands Bidi Agents and
Amazon Nova Sonic. Made in Joburg by Katlego.
