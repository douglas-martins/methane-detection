# Pi project configuration

This directory mirrors the repository's Claude Code subagents for Pi.
The project-local subagent extension is based on Pi's official
`examples/extensions/subagent` implementation.

## Model mapping

Only authenticated OpenAI Codex models are enabled for this project:

| Claude tier | Pi model |
| --- | --- |
| `haiku` | `openai-codex/gpt-5.6-luna` |
| `sonnet` | `openai-codex/gpt-5.6-terra` |

The mapping preserves the intent of the Claude configuration: Luna handles
lower-cost mechanical checks, while Terra handles engineering-judgment work.

## Usage

Trust the repository when Pi prompts so it can load `.pi` resources. Then ask
Pi to invoke an agent through the `subagent` tool with
`agentScope: "project"`. The extension supports single, parallel, and chained
execution.

After changing an agent or extension, run `/reload` in an existing Pi session.
