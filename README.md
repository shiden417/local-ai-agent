# local-ai-agent

Ollama + Qwen3:8B + LiteLLM を使ったローカルAI Agentです。

## Architecture

```
User
  ↓
Agent Runtime
  ↓
LiteLLM
  ↓
Ollama / Qwen3:8B
  ↓
Tool selection
  ↓
PowerShell tool
  ↓
Observe result
  ↓
Qwen3 decides next step
```

## Requirements

- Windows
- Python 3.12+
- Ollama
- Qwen3:8B

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Agentを操作したい作業ディレクトリで起動します。

```powershell
python agent.py
```

Agent Runtimeは起動時のカレントディレクトリを作業ディレクトリとして固定します。

## Current tool

- execute_command

## Roadmap

1. execute_command
2. list_directory
3. read_file
4. search_files
5. write_file / edit_file
6. Git tools
7. safety / confirmation
8. autonomous task loop
9. structured tool protocol
