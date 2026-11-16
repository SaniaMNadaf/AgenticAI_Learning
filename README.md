# Agentic AI Learning

Learning exercises are organized under `workshop/Day1` through `workshop/Day6`.

## Setup

From the repository root, install the Day 6 dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set `GEMINI_API_KEY` in your environment for Gemini-powered exercises. Keep the
key in an ignored `.env` file or environment variable; do not commit it.

## Exercises

- **Day 1:** Run `python workshop\Day1\PromptRelay.py` for the four-stage
  meeting-notes prompt comparison. Other Day 1 scripts demonstrate prompting
  and tool calling.
- **Day 2:** Put source PDFs in `workshop\Day2\sample_corpus`. From
  `workshop\Day2`, install/start Ollama and pull `nomic-embed-text`, then run
  `python p4.py --input sample_corpus --backend chroma --embedder ollama`.
- **Day 3:** From `workshop\Day3`, run
  `python D3P3.py --question "What is the fine for a late library book?"`.
  Query rewriting is optional; enable it with `--rewrite on`.
- **Day 4 and Day 5:** Run the corresponding scripts from their day folders.
- **Day 6 API:** From `workshop\Day6`, run
  `python -m uvicorn main:app --port 8000`. Check `http://localhost:8000/health`
  and POST `{"question":"What is the fine for a late library book?"}` to
  `http://localhost:8000/ask`.
- **Day 6 UI:** In a second terminal, from `workshop\Day6`, run
  `python -m streamlit run app.py`; the API must already be running.
- **Trace viewer:** From `workshop\Day6`, run `python trace_viewer.py` after the
  API has written events to `agent.log`.

The Day 6 runtime dependencies are listed in
[workshop/Day6/requirements.txt](workshop/Day6/requirements.txt). The root
requirements file includes that list for convenience.

## Deployment

Vercel deployment has not been verified. Docker was skipped for this local
learning setup; container deployment and cloud credentials are outside the
current local run-through.
