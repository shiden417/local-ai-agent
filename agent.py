from pathlib import Path

from agent.runtime import AgentRuntime


def main() -> None:
    runtime = AgentRuntime(
        working_directory=Path.cwd(),
        max_iterations=10,
    )

    print("Local AI Agent")
    print(f"Working Directory: {runtime.working_directory}")
    print("exit または quit で終了します。")

    while True:
        try:
            user_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.strip().lower() in {"exit", "quit"}:
            break

        if not user_input.strip():
            continue

        try:
            print(runtime.run(user_input))
        except Exception as exc:
            print(f"Agent error: {exc}")


if __name__ == "__main__":
    main()
