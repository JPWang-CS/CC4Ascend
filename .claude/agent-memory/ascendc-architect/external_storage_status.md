---
name: external-storage-status
description: Tracks setup status of external memory and skills directories that the agent depends on
metadata:
  type: project
---

# External Storage Status

## Memory Directory
- **Path**: D:\desktop\牛马\AI-GEN\Memory\AscendC-expert\
- **Status**: DOES NOT EXIST -- needs manual creation (mkdir denied for bash)
- **Purpose**: Persistent store for discovered design patterns, common pitfalls, effective tiling configurations, performance benchmarks
- **Action needed**: User must create this directory manually

## Skills Directory
- **Path**: D:\desktop\牛马\AI-GEN\skills\
- **Status**: DOES NOT EXIST -- needs manual creation (mkdir denied for bash)
- **Purpose**: Internal design standards, best practices, chip specifications, project-specific constraints
- **Action needed**: User must create this directory and populate with skill documents

## Project Agent Memory
- **Path**: D:/Desktop/代码/Claude-AutoGen/CC4Ascend/.claude/agent-memory/ascendc-architect/
- **Status**: Created and initialized (2026-05-21) with MEMORY.md + 3 memory files
