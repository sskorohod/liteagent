---
name: audio_insight_ru
description: "Audio Insight RU — local analysis of Russian audio files: transcription (STT), speaker diarization, NER annotation, RAG export."
metadata:
  emoji: "🎧"
  keywords:
    - аудиоанализ
    - анализ аудио
    - транскрипция
    - транскрибация
    - диаризация
    - diarization
    - audio-insight
    - audio_insight
    - audio insight
    - голосовой анализ
    - анализируй аудио
    - анализируй запись
    - распознай спикеров
    - извлеки термины
    - аннотируй аудио
    - funasr
    - whisperx
    - анализ голоса
    - ogg анализ
    - mp3 анализ
    - wav анализ
  tools:
    - transcribe_voice
    - exec_command
    - write_file
    - read_file
    - kb_ingest
---

## Audio Insight RU (activated)

You are equipped with a full workflow for local analysis of Russian-language audio files. All processing is local — no audio is sent to the cloud.

### Workflow: Audio File → Structured Knowledge

**Step 1 — Transcription (STT)**
Use the built-in `transcribe_voice` tool first (OpenAI Whisper/Deepgram/Groq).
- Language hint: always pass `language="ru"` when available.
- If the file is large (>25 MB), split it first with `exec_command`:
  ```bash
  ffmpeg -i input.ogg -f segment -segment_time 300 -c copy chunk_%03d.ogg
  ```

**Step 2 — Diarization (speaker detection)**
After transcription, if multiple speakers are suspected, run diarization via `exec_command`:
```bash
python3 -c "
import json, sys
# Requires: pip install pyannote.audio (one-time)
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
diarization = pipeline('AUDIO_FILE')
segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({'start': round(turn.start,2), 'end': round(turn.end,2), 'speaker': speaker})
print(json.dumps(segments, ensure_ascii=False))
"
```
If pyannote is not available, fall back to annotating the transcript with `[Speaker A]`/`[Speaker B]` based on paragraph breaks.

**Step 3 — NER (entity extraction)**
Extract key entities from the transcript using `exec_command` with a local Ollama call:
```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:latest",
  "prompt": "Extract named entities (persons, dates, organizations, locations, key terms) from this Russian text. Return JSON array: [{\"entity\", \"type\", \"context\"}].\n\nText: TRANSCRIPT_HERE",
  "stream": false
}'
```
Types to extract: PERSON, ORG, DATE, LOCATION, TERM, TOPIC.

**Step 4 — Summary + key questions**
Generate a concise summary and 3-5 key questions for RAG indexing using Ollama:
```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:latest",
  "prompt": "Summarize this Russian audio transcript in 3-5 sentences. Then list 5 key questions a search engine should answer based on this content. Format: {\"summary\": \"...\", \"questions\": [...]}.\n\nTranscript: TRANSCRIPT_HERE",
  "stream": false
}'
```

**Step 5 — Save to RAG / Knowledge Base**
Save the annotated result as a structured markdown file and ingest it:
```
# Audio Analysis: <filename>
Date: <timestamp>
Duration: <seconds>s

## Transcript
<full transcript with speaker labels>

## Key Entities
- PERSON: ...
- ORG: ...
- DATE: ...

## Summary
<summary>

## Key Questions
1. ...
```
Then call `kb_ingest` with the saved `.md` file path to add it to the Knowledge Base.

Alternatively use `write_file` to save to `~/rag/audio/<filename>.md` for manual RAG indexing.

### CRITICAL RULES:
1. **Always use `language="ru"` for STT** — dramatically improves Russian accuracy.
2. **Never send audio to cloud without explicit user confirmation.** Prefer local Ollama for NER/summary.
3. **If a tool/library is missing** — detect it with `exec_command` (`which ffmpeg`, `pip show pyannote.audio`) and report what needs to be installed before proceeding.
4. **Diarization is optional** — only run if user asks for speaker separation or if transcript seems to contain multiple voices.
5. **Always save results** — never just print the transcript. Write to file so it persists.
6. **Check file format first** — supported: `.ogg`, `.mp3`, `.wav`, `.m4a`, `.webm`. Convert with ffmpeg if needed.

### Quick checks before starting:
```bash
which ffmpeg && echo "ffmpeg OK" || echo "ffmpeg MISSING — install: brew install ffmpeg"
curl -s http://localhost:11434/api/tags | python3 -c "import json,sys; models=[m['name'] for m in json.load(sys.stdin)['models']]; print('Ollama models:', models)" 2>/dev/null || echo "Ollama not running"
python3 -c "import pyannote.audio; print('pyannote OK')" 2>/dev/null || echo "pyannote not installed (optional)"
```

### Supported audio workflows:
- **Single voice message** (Telegram .ogg) → transcribe → entities → save
- **Meeting recording** → transcribe → diarize by speaker → summary per speaker → save
- **Long interview** → chunk → transcribe each → merge → NER + summary → kb_ingest
- **Voice memo** → transcribe → extract action items → save as task notes
