# Astra planner and Sol implementer example

Copy `AGENTS.md` and the `.codex` directory to the root of a trusted Codex project. Start Codex from that project and verify the effective model, reasoning effort, feature state, and available implementer role in your installed client before assigning work.

```sh
cd /path/to/your/trusted-project
codex
```

The configuration is verified against the Codex source snapshot at commit `944d6fd1ba4baab69dbedd205282dc72ec20abb5`. It was not tested through a fresh authenticated invocation. Model and multi-agent availability depend on the installed client, effective configuration, and account access.

Runtime, UI, or CLI overrides can supersede project model settings. Managed requirements constrain which settings are allowed; ordinary system configuration has lower priority than those requirements. The `AGENTS.md` file guides the root agent but does not remove its write-capable tools. Enforce a strict planner-only boundary in the host application if that is a hard requirement.
