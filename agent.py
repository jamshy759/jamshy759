"""Personal AI assistant agent (Claude API + tool runner).

Run:  python agent.py
Needs ANTHROPIC_API_KEY in the environment (or `ant auth login`).
"""
import argparse
import json
from datetime import date, datetime
from pathlib import Path

import anthropic
from anthropic import beta_tool

MODEL = "claude-opus-5-5"
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
TASKS_FILE = DATA / "tasks.json"
NOTES_FILE = DATA / "notes.json"
HISTORY_FILE = DATA / "history.json"
WORKSPACE = DATA / "workspace"
WORKSPACE.mkdir(exist_ok=True)

SYSTEM = f"""You are a personal assistant agent for your user. Today is {date.today():%A, %Y-%m-%d}.
You can manage their task list, keep notes, and search the web.
- Use tools to act; do not just describe what you would do.
- When the user mentions something to do, offer to add it as a task (add it directly if they clearly asked).
- Save notes when the user says to remember something.
- You can read/write files in the workspace folder for drafts, plans and summaries.
- Be concise. Confirm what you changed after using a tool."""


def _load(path: Path) -> list:
    return json.loads(path.read_text()) if path.exists() else []


def _save(path: Path, items: list) -> None:
    path.write_text(json.dumps(items, indent=2))


@beta_tool
def add_task(title: str, due: str = "") -> str:
    """Add a task to the to-do list.

    Args:
        title: What needs to be done.
        due: Optional due date in YYYY-MM-DD format.
    """
    tasks = _load(TASKS_FILE)
    task_id = max((t["id"] for t in tasks), default=0) + 1
    tasks.append({"id": task_id, "title": title, "due": due, "done": False})
    _save(TASKS_FILE, tasks)
    return f"Added task #{task_id}: {title}"


@beta_tool
def list_tasks(include_done: bool = False) -> str:
    """List tasks on the to-do list.

    Args:
        include_done: Also show completed tasks.
    """
    tasks = [t for t in _load(TASKS_FILE) if include_done or not t["done"]]
    if not tasks:
        return "No tasks."
    return "\n".join(
        f"#{t['id']} [{'x' if t['done'] else ' '}] {t['title']}"
        + (f" (due {t['due']})" if t["due"] else "")
        for t in tasks
    )


@beta_tool
def complete_task(task_id: int) -> str:
    """Mark a task as done.

    Args:
        task_id: The numeric id of the task.
    """
    tasks = _load(TASKS_FILE)
    for t in tasks:
        if t["id"] == task_id:
            t["done"] = True
            _save(TASKS_FILE, tasks)
            return f"Completed task #{task_id}: {t['title']}"
    return f"No task with id {task_id}."


@beta_tool
def save_note(text: str) -> str:
    """Save a note for the user to remember later.

    Args:
        text: The note content.
    """
    notes = _load(NOTES_FILE)
    notes.append({"time": datetime.now().isoformat(timespec="minutes"), "text": text})
    _save(NOTES_FILE, notes)
    return "Note saved."


@beta_tool
def search_notes(query: str) -> str:
    """Search saved notes (case-insensitive substring match).

    Args:
        query: Text to look for.
    """
    hits = [n for n in _load(NOTES_FILE) if query.lower() in n["text"].lower()]
    if not hits:
        return "No matching notes."
    return "\n".join(f"- [{n['time']}] {n['text']}" for n in hits)


@beta_tool
def due_soon(days: int = 7) -> str:
    """Show overdue tasks and tasks due within the next N days.

    Args:
        days: How many days ahead to look.
    """
    today = date.today()
    out = []
    for t in _load(TASKS_FILE):
        if t["done"] or not t["due"]:
            continue
        delta = (date.fromisoformat(t["due"]) - today).days
        if delta < 0:
            out.append(f"OVERDUE #{t['id']} {t['title']} (was due {t['due']})")
        elif delta <= days:
            out.append(f"#{t['id']} {t['title']} (due {t['due']}, in {delta}d)")
    return "\n".join(out) or "Nothing overdue or due soon."


def _safe(name: str) -> Path:
    p = (WORKSPACE / name).resolve()
    if WORKSPACE.resolve() not in p.parents and p != WORKSPACE.resolve():
        raise ValueError("Path must stay inside the workspace folder.")
    return p


@beta_tool
def list_files() -> str:
    """List files in the agent's workspace folder (data/workspace)."""
    files = [str(f.relative_to(WORKSPACE)) for f in WORKSPACE.rglob("*") if f.is_file()]
    return "\n".join(sorted(files)) or "Workspace is empty."


@beta_tool
def read_file(name: str) -> str:
    """Read a text file from the workspace folder.

    Args:
        name: File path relative to the workspace.
    """
    try:
        return _safe(name).read_text()[:50000]
    except (OSError, ValueError) as e:
        return f"Error: {e}"


@beta_tool
def write_file(name: str, content: str) -> str:
    """Create or overwrite a text file in the workspace (drafts, summaries, plans).

    Args:
        name: File path relative to the workspace.
        content: Full file contents.
    """
    try:
        p = _safe(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {name}"
    except (OSError, ValueError) as e:
        return f"Error: {e}"


TOOLS = [
    add_task, list_tasks, complete_task, save_note, search_notes,
    due_soon, list_files, read_file, write_file,
    {"type": "web_search_20260209", "name": "web_search"},  # runs on Anthropic's servers
]


BRIEFING_PROMPT = (
    "Give me my morning briefing: overdue and upcoming tasks, anything notable in my notes, "
    "and a suggested top-3 priorities for today. Keep it short."
)


def ask(client, history: list) -> str:
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        tools=TOOLS,
        messages=history,
    )
    final = runner.until_done()
    return "".join(b.text for b in final.content if b.type == "text") or "(done)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--briefing", action="store_true", help="print a morning briefing and exit")
    ap.add_argument("--fresh", action="store_true", help="start without saved conversation history")
    args = ap.parse_args()

    client = anthropic.Anthropic()
    if args.briefing:
        print(ask(client, [{"role": "user", "content": BRIEFING_PROMPT}]))
        return

    history = [] if args.fresh else _load(HISTORY_FILE)
    print("Personal assistant ready. Type 'exit' to quit, 'briefing' for your morning briefing.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        if user.lower() == "briefing":
            user = BRIEFING_PROMPT
        history.append({"role": "user", "content": user})
        try:
            text = ask(client, history)
        except anthropic.APIError as e:
            history.pop()
            print(f"[API error: {e}]")
            continue
        print(f"\nagent> {text}")
        # Keep only final text; tool results are already persisted in data/.
        history.append({"role": "assistant", "content": text})
        _save(HISTORY_FILE, history[-40:])


if __name__ == "__main__":
    main()
