"""Personal AI assistant agent (Claude API + tool runner).

Run:  python agent.py
Needs ANTHROPIC_API_KEY in the environment (or `ant auth login`).
"""
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

SYSTEM = f"""You are a personal assistant agent for your user. Today is {date.today():%A, %Y-%m-%d}.
You can manage their task list, keep notes, and search the web.
- Use tools to act; do not just describe what you would do.
- When the user mentions something to do, offer to add it as a task (add it directly if they clearly asked).
- Save notes when the user says to remember something.
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


TOOLS = [
    add_task, list_tasks, complete_task, save_note, search_notes,
    {"type": "web_search_20260209", "name": "web_search"},  # runs on Anthropic's servers
]


def main() -> None:
    client = anthropic.Anthropic()
    history: list[dict] = []
    print("Personal assistant ready. Type 'exit' to quit.")
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        try:
            runner = client.beta.messages.tool_runner(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM,
                thinking={"type": "adaptive"},
                tools=TOOLS,
                messages=history,
            )
            final = runner.until_done()
        except anthropic.APIError as e:
            history.pop()
            print(f"[API error: {e}]")
            continue
        text = "".join(b.text for b in final.content if b.type == "text")
        print(f"\nagent> {text}")
        # Keep only the final text in history; tool results are already persisted in data/.
        history.append({"role": "assistant", "content": text or "(done)"})


if __name__ == "__main__":
    main()
