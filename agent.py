from pathlib import Path

from agent.safety import SafetyPolicy
from agent.llm import MODEL
from agent.runtime import AgentRuntime
from agent.trace import TraceRecorder
from agent.terminal_ui import TerminalUI


def print_tasks(runtime: AgentRuntime) -> None:
    tasks = runtime.list_tasks()
    if not tasks:
        runtime.terminal_ui.info("Taskはありません。") if runtime.terminal_ui else print("Taskはありません。")
        return

    print("\nTasks:")
    for task in tasks:
        print(
            f"- {task.task_id} | "
            f"{task.status.value} | "
            f"{task.goal[:80]}"
        )


def print_stats(trace: TraceRecorder) -> None:
    metrics = trace.summary()
    print("\nAgent statistics:")
    print(f"- LLM calls: {metrics['llm_calls']}")
    print(f"- Tool calls: {metrics['tool_calls']}")
    print(f"- Avg LLM latency: {metrics['avg_llm_duration_ms']} ms")
    print(f"- Avg Tool latency: {metrics['avg_tool_duration_ms']} ms")
    print(f"- Prompt chars: {metrics['prompt_chars']}")
    print(f"- Tool schema chars: {metrics['tool_schema_chars']}")
    print(f"- Prompt tokens: {metrics['prompt_tokens']}")
    print(f"- Completion tokens: {metrics['completion_tokens']}")
    print(f"- Reasoning tokens: {metrics['reasoning_tokens']}")
    print(f"- Trace: {trace.path}")

def print_permissions(policy: SafetyPolicy) -> None:
    entries = policy.entries()
    if not entries:
        print("\nLearned permissions: none")
        return

    print("\nLearned permissions:")
    for entry in entries:
        description = entry.get("description", "")
        print(f"- {entry['key']} | {description[:100]}")


def main() -> None:
    ui = TerminalUI(MODEL)
    trace = TraceRecorder()
    runtime = AgentRuntime(
        working_directory=Path.cwd(),
        max_iterations=10,
        terminal_ui=ui,
        trace_recorder=trace,
    )

    ui.startup(
        str(runtime.working_directory),
        f"Auto / LM Studio (learned approvals: {len(runtime.safety.entries())} rules)",
    )

    while True:
        try:
            user_input = input("You > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        command = user_input.strip()
        lowered = command.lower()
        if lowered in {"exit", "quit", "/exit", "/quit"}:
            print("終了します。")
            break

        if lowered == "/tasks":
            print_tasks(runtime)
            continue

        if lowered == "/permissions":
            print_permissions(runtime.safety)
            continue

        if lowered in {"/stats", "/metrics"}:
            print_stats(trace)
            continue

        if lowered in {"/clear-permissions", "/clear-approvals"}:
            runtime.safety.clear()
            print("Learned permissionsをクリアしました。")
            continue

        if lowered in {"/clear-context", "/clear-session"}:
            runtime.clear_session_context()
            print("Session Contextをクリアしました。")
            continue

        if not command:
            continue

        try:
            runtime.run(command)
        except Exception as exc:
            ui.error(f"Agent error: {exc}")


if __name__ == "__main__":
    main()
