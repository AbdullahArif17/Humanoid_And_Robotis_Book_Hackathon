---
id: 10
title: Empty Feature Description for Specify Command
stage: general
date: 2025-12-05
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: 0-book-project-plan
user: user
command: /sp.specify
labels: ["spec", "error", "empty-argument"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files: null
tests: null
---

## Prompt

/sp.specify

## Response snapshot

ERROR: No feature description provided for /sp.specify. Please provide a detailed description of the feature you want to specify.

## Outcome

- ✅ Impact: Failed to create specification due to empty feature description.
- 🧪 Tests: N/A
- 📁 Files: No files created.
- 🔁 Next prompts: Provide a detailed feature description for /sp.specify.
- 🧠 Reflection: The user invoked /sp.specify with empty arguments. The system correctly identified this as an error and reported it. A PHR is being created to document this interaction.

## Evaluation notes (flywheel)

- Failure modes observed: Empty feature description provided.
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): Consider adding a more user-friendly prompt for empty /sp.specify calls.
