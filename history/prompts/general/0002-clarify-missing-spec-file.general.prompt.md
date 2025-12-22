---
id: 0002
title: Clarify Missing Spec File
stage: general
date: 2025-12-07
surface: agent
model: claude-sonnet-4-5-20251101
feature: none
branch: 0-book-project-plan
user: user
command: /sp.clarify
labels: ["clarification", "error"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:

tests:

---

## Prompt

$ARGUMENTS

## Response snapshot

The spec file for this feature is missing. Please run `/sp.specify` first to create the feature specification.

## Outcome

- ✅ Impact: N/A
- 🧪 Tests: N/A
- 📁 Files: N/A
- 🔁 Next prompts: N/A
- 🧠 Reflection: User needs to create a spec file before clarification can proceed.

## Evaluation notes (flywheel)

- Failure modes observed: Missing spec file.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
