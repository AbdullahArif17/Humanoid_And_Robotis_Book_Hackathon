---
id: 1
title: Update Constitution for AI-native Software Development Research Paper
stage: constitution
date: 2025-12-04
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution Role: Expert Academic Research Assistant for AI-native Software Development Objective: Generate a rigorous, academically precise, and reproducible research paper on AI-native software development following the core principles and constraints outlined below. Input Constitution: Core Principles: Accuracy: Verify all claims using primary sources. Clarity: Academic tone for a computer science audience. Reproducibility: All sources cited and traceable. Rigor: Prefer peer-reviewed publications. Key Standards: All factual claims must be traceable. Citation format: APA style. Source types: Minimum 50% peer-reviewed articles. Plagiarism: 0% tolerance. Writing clarity: Flesch-Kincaid grade 10–12. Constraints: Word count: 5,000–7,000 words. Minimum 15 sources. Deliverable format: PDF with embedded citations. Success Criteria: Every claim verified. Zero plagiarism. Passes fact-checking review. Instructions to Claude Code: Generate a detailed outline first, breaking the paper into sections (Introduction, Background, Methods, Results, Discussion, Conclusion, References). For each section: Draft text with academic tone. Include in-text APA citations for all factual statements. Ensure at least 50% of references are peer-reviewed. Provide a bibliography in APA style, automatically formatted. Highlight source links for each claim to allow verification. Perform a plagiarism check on the generated text before finalizing. Ensure total word count is within 5,000–7,000 words. Output the result in a structured Markdown or HTML format suitable for PDF conversion. Optional Advanced Step (Claude Code Subagent / Skill): Use a “Fact-Checker” subagent that cross-verifies all claims against online scholarly databases (Google Scholar, IEEE Xplore, arXiv). Use a “Citation Formatter” subagent to automatically generate APA-style references.
labels: ["constitution", "update"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - .specify/memory/constitution.md
tests: null
---

## Prompt

Role: Expert Academic Research Assistant for AI-native Software Development

Objective: Generate a rigorous, academically precise, and reproducible research paper on AI-native software development following the core principles and constraints outlined below.

Input Constitution:

Core Principles:

Accuracy: Verify all claims using primary sources.

Clarity: Academic tone for a computer science audience.

Reproducibility: All sources cited and traceable.

Rigor: Prefer peer-reviewed publications.

Key Standards:

All factual claims must be traceable.

Citation format: APA style.

Source types: Minimum 50% peer-reviewed articles.

Plagiarism: 0% tolerance.

Writing clarity: Flesch-Kincaid grade 10–12.

Constraints:

Word count: 5,000–7,000 words.

Minimum 15 sources.

Deliverable format: PDF with embedded citations.

Success Criteria:

Every claim verified.

Zero plagiarism.

Passes fact-checking review.

Instructions to Claude Code:

Generate a detailed outline first, breaking the paper into sections (Introduction, Background, Methods, Results, Discussion, Conclusion, References).

For each section:

Draft text with academic tone.

Include in-text APA citations for all factual statements.

Ensure at least 50% of references are peer-reviewed.

Provide a bibliography in APA style, automatically formatted.

Highlight source links for each claim to allow verification.

Perform a plagiarism check on the generated text before finalizing.

Ensure total word count is within 5,000–7,000 words.

Output the result in a structured Markdown or HTML format suitable for PDF conversion.

Optional Advanced Step (Claude Code Subagent / Skill):

Use a “Fact-Checker” subagent that cross-verifies all claims against online scholarly databases (Google Scholar, IEEE Xplore, arXiv).

Use a “Citation Formatter” subagent to automatically generate APA-style references.

## Response snapshot

The project constitution has been successfully updated to version 1.0.0, reflecting the specific requirements for an "AI-native Software Development Research Paper." This is a major version bump due to the complete transformation from a generic template to a specialized one, incorporating new principles, standards, and instructions.

The following files require manual follow-up:
- `.specify/templates/plan-template.md`: Generic, needs adaptation for the research paper context.
- `.specify/templates/spec-template.md`: Generic, needs adaptation for the research paper context.
- `.specify/templates/tasks-template.md`: Generic, needs adaptation for the research paper context.
- `.specify/templates/commands/sp.constitution.md`: This file was not found and may need creation or clarification regarding its purpose in this project.
- `README.md`: This file was not found and may need creation to reflect project details.
- `docs/quickstart.md`: This file was not found and may need creation for quick project onboarding.

Suggested commit message:
`docs: amend constitution to v1.0.0 (initial research paper principles + governance)`

## Outcome

- ✅ Impact: Constitution updated for AI-native Software Development Research Paper
- 🧪 Tests: N/A
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: Review and adapt templates for research paper context, create missing documentation files.
- 🧠 Reflection: Successfully updated the project constitution based on user input, adapting a generic template to a specific academic research paper context. Encountered missing script for PHR creation, handled with agent-native fallback. Identified several templates and documentation files requiring further adaptation or creation.

## Evaluation notes (flywheel)

- Failure modes observed: Script for PHR creation not found.
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): Ensure PHR creation script is available or improve agent-native fallback.
