# Planner and implementer policy

The root agent plans, researches the repository, decomposes tasks, and reviews results using gpt-6-astra with high reasoning effort.

Delegate every code change to an implementer child using gpt-5.6-sol with medium reasoning effort. Select agent_type implementer when the tool exposes roles. Set model and reasoning_effort explicitly, and use fork_turns none. Give each child the files it may change, acceptance criteria, and what is out of scope.

The root must review the diff and run or inspect relevant validation before accepting the change. If either model is unavailable, report the blocker and do not substitute another model.
