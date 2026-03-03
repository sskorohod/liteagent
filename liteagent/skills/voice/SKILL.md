---
name: voice
description: "Voice configuration (TTS/STT). Use for: changing voice, provider, presets, testing speech synthesis, transcription settings."
metadata:
  emoji: "🎙️"
  keywords:
    - голос
    - голосов
    - голосом
    - голоса
    - озвуч
    - озвучк
    - озвучив
    - речь
    - речи
    - речью
    - произнес
    - произнос
    - произношен
    - tts
    - stt
    - text-to-speech
    - speech-to-text
    - voice
    - voices
    - whisper
    - elevenlabs
    - edge tts
    - edge-tts
    - пресет
    - preset
    - presets
    - пресеты
    - распознав
    - транскриб
    - transcri
    - только текст
    - text only
    - без озвучк
    - без звук
    - mute
  tools:
    - get_voice_settings
    - set_voice_settings
    - list_voice_providers
    - test_tts
    - save_voice_preset
    - load_voice_preset
---

## Voice configuration skill (activated)
You have voice tools to configure text-to-speech and speech-to-text settings:
- **get_voice_settings** — check current voice configuration, provider status, and pricing
- **set_voice_settings** — change TTS provider/voice/model/auto-mode, STT provider/model/language, ElevenLabs parameters
- **list_voice_providers** — see all available providers with models, voices, pricing, and API key status
- **test_tts** — generate audio from text (optionally override voice/provider without changing persistent settings)
- **save_voice_preset** / **load_voice_preset** — save/load named voice profiles for quick switching

When the user asks to change voice, enable/disable TTS, adjust speech settings, or test a voice — use these tools.
Built-in presets: professional, casual, storyteller, fast_free, russian.
Cost: OpenAI ~$15/1M chars, ElevenLabs ~$30/1M, Edge TTS — free (unlimited).

## Communication mode switching
**IMPORTANT: When user requests a communication mode change, you MUST call the `set_voice_settings` tool — do NOT just reply with text. The setting must be actually changed, not just acknowledged.**

- **Text only** ("только текстом", "без голоса", "без озвучки", "mute") → MUST call `set_voice_settings(tts_auto="off")`
- **Voice + text** ("голос и текст", "озвучивай", "дублируй голосом") → MUST call `set_voice_settings(tts_auto="always")`
- **Voice replies to voice input** ("отвечай голосом на голос") → MUST call `set_voice_settings(tts_auto="inbound")`

After calling the tool, confirm the change to the user briefly.
