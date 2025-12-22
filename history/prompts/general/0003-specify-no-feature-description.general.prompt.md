---
id: 0003
title: Specify No Feature Description
stage: general
date: 2025-12-07
surface: agent
model: claude-sonnet-4-5-20251101
feature: none
branch: 0-book-project-plan
user: user
command: /sp.specify
labels: ["specify", "error"]
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

No feature description provided.

## Outcome

- ✅ Impact: N/A
- 🧪 Tests: N/A
- 📁 Files: N/A
- 🔁 Next prompts: User needs to provide a feature description for /sp.specify.
- 🧠 Reflection: The /sp.specify command requires a feature description as an argument.

## Evaluation notes (flywheel)

- Failure modes observed: Missing feature description.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
