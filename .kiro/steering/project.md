# Jozi Buddy — project facts

- **What:** a local voice AI agent. It runs on the user's laptop, listens through the microphone and
  talks back through the speakers in near real time. Not a web app, not deployed to AWS compute.
- **Built with:** Strands Agents Bidi API (GA `strands.bidi` imports, not `strands.experimental.bidi`),
  installed as `strands-agents[bidi-all,bidi-pyaudio]`. `AudioIO(audio_processor=True)` for echo
  cancellation.
- **Model:** Amazon Nova 2.5 Sonic, model id `amazon.nova-2-5-sonic`, on Amazon Bedrock.
- **Region and credentials:** AWS profile `jozi-buddy` (region `eu-north-1`), signed in with
  `aws login`. No long-lived access keys. The tutorial uses `us-east-1` with an IAM access key; both
  regions host the model, and this project deliberately uses short-lived `aws login` credentials.
- **Personality:** "Jozi Buddy", a friendly Johannesburg-based buddy who chats casually and recommends
  things to do in the city. Spoken replies are one or two sentences.
- **Grounding:** never invent Johannesburg places, opening hours, prices, directions or safety advice.
  If it doesn't know, it says so.
- **Publishing:** code on GitHub; a static landing page in `docs/` on GitHub Pages.
- **Source of steps:** the Mzansi AWS Community Day tutorial (a local copy, `TUTORIAL.md`, is kept
  out of git).
