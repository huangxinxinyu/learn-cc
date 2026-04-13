import os
import json
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion

load_dotenv(override=True)

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
MODEL = os.getenv("LITELLM_MODEL", "dashscope/qwen-plus")

WORKDIR = Path.cwd()


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return f"Error: Dangerous command blocked"
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKDIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def run_read(path: str, limit: int = None) -> str:
    try:
        fp = safe_path(path)
        text = fp.read_text()
        lines = text.splitlines()
        
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw.get("file_path") or kw.get("path"), kw.get("limit")),
    "write_file": lambda **kw: run_write(kw.get("file_path") or kw.get("path"), kw.get("content")),
    "edit_file":  lambda **kw: run_edit(kw.get("file_path") or kw.get("path"), kw.get("old_text"), kw.get("new_text")),
}


SYSTEM_PROMPT = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
]


def agent_loop(messages: list):
    while True:
        response = completion(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            tools=TOOLS,
            max_tokens=4000,
        )
        
        assistant_message = response.choices[0].message
        messages.append({"role": "assistant", "content": assistant_message.content})
        
        if not assistant_message.tool_calls:
            return
        
        tool_results = []
        for block in assistant_message.tool_calls:
            handler = TOOL_HANDLERS.get(block.function.name)
            args = block.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            print(f"\033[35m[DEBUG] {block.function.name}: {args}\033[0m")
            output = handler(**args) if handler else f"Unknown tool: {block.function.name}"
            print(f"\033[33m> {block.function.name}\033[0m")
            print(output[:200] if len(output) > 200 else output)
            tool_results.append({
                "tool_call_id": block.id,
                "role": "tool",
                "name": block.function.name,
                "content": output
            })
        
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result["tool_call_id"],
                "content": result["content"]
            })


def main():
    history = []
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        
        if query.strip().lower() in ("q", "exit", ""):
            break
        
        history.append({"role": "user", "content": query})
        agent_loop(history)
        
        final_message = history[-1]
        final_text = final_message.get("content", "")
        if final_text:
            print(final_text)
        print()


if __name__ == "__main__":
    main()
