from pathlib import Path

from agent.runtime import AgentRuntime


def print_tasks(runtime: AgentRuntime) -> None:
    tasks = runtime.list_tasks()
    if not tasks:
        print("Taskはありません。")
        return

    print("\nTasks:")
    for task in tasks:
        print(
            f"- {task.task_id} | "
            f"{task.status.value} | "
            f"{task.goal[:80]}"
        )


def main() -> None:
    runtime = AgentRuntime(
        working_directory=Path.cwd(),
        max_iterations=10,
    )

    print("Local AI Agent")
    print(f"Working Directory: {runtime.working_directory}")
    print("exit または quit で終了します。")
    print("/tasks でTask一覧を表示できます。")

    while True:
        try:
            user_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        command = user_input.strip()
        if command.lower() in {"exit", "quit"}:
            break

        if command.lower() == "/tasks":
            print_tasks(runtime)
            continue

        if not command:
            continue

        try:
            print(runtime.run(command))
        except Exception as exc:
            print(f"Agent error: {exc}")


if __name__ == "__main__":
    main()
