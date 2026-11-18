# Personal AI Assistant

A command-line AI agent built on the Claude API. It keeps a to-do list and notes
(stored locally in `data/`) and can search the web.

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key
python agent.py
```

## Try
- "Add a task to renew my passport by 2026-12-31"
- "What's on my list?"
- "Remember that my dentist is Dr. Khan"
- "Search the web for the best budget laptops this month"

## Extend
Add a function decorated with `@beta_tool` in `agent.py` and put it in `TOOLS`.
Ideas: Gmail/Calendar access, job search, reading local files.
