---
id: 2
title: Update Constitution for Physical AI & Humanoid Robotics Book
stage: constitution
date: 2025-12-04
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution Role: Expert Author and AI Assistant for Physical AI & Humanoid Robotics Objective: Generate a comprehensive, academically rigorous, and structured book on Physical AI & Humanoid Robotics suitable for students in a technical course. Include simulation, ROS 2, NVIDIA Isaac, VLA, and humanoid robotics content, integrating real-world and simulated robotics concepts. Input Reference: Course outline, modules, weekly breakdown, hardware requirements, lab options, and capstone project description (as provided). Emphasis on Physical AI principles, embodied intelligence, humanoid robot design, simulation, AI perception, and Vision-Language-Action integration. Constraints: Book length: 50–80 pages (approx. 20,000–40,000 words). Structured in chapters corresponding to modules and weekly topics. Include diagrams, tables, and code snippets (ROS 2, Gazebo, Isaac, VLA). Writing clarity: Flesch-Kincaid grade 10–12. All technical claims must be accurate and reproducible; cite sources where appropriate (APA style). Output in Markdown or Docusaurus-compatible format, ready for GitHub Pages deployment. Instructions to Claude Code: Generate a table of contents with chapters for each module and week. For each chapter: Include theoretical explanation (Physical AI, robotics principles). Provide practical implementation examples: ROS 2 code snippets, Gazebo simulation steps, Isaac Sim instructions, Unity visualization tips. Include hardware guidance: Jetson, RealSense, Edge Kits, and optional robot options. Add diagrams, flowcharts, or tables to explain complex systems. Add a capstone project chapter: Autonomous humanoid performing VLA tasks, voice-to-action, navigation, manipulation. Include FAQs, troubleshooting tips, and learning outcomes per chapter. Ensure RAG-readiness: Chunk text into retrievable sections for embedding in a chatbot. Provide a reference section with URLs or papers for all factual claims. Optional Advanced Step (Claude Code Subagents / Skills): “Code Snippet Generator”: Automatically produce ROS 2, Gazebo, Isaac, or VLA example code blocks. “Diagram Generator”: Suggest diagrams for complex robotic systems. “RAG Sectioner”: Split book into retrievable sections for chatbot integration.
labels: ["constitution", "update", "book", "robotics"]
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

Role: Expert Author and AI Assistant for Physical AI & Humanoid Robotics

Objective: Generate a comprehensive, academically rigorous, and structured book on Physical AI & Humanoid Robotics suitable for students in a technical course. Include simulation, ROS 2, NVIDIA Isaac, VLA, and humanoid robotics content, integrating real-world and simulated robotics concepts.

Input Reference:

Course outline, modules, weekly breakdown, hardware requirements, lab options, and capstone project description (as provided).

Emphasis on Physical AI principles, embodied intelligence, humanoid robot design, simulation, AI perception, and Vision-Language-Action integration.

Constraints:

Book length: 50–80 pages (approx. 20,000–40,000 words).

Structured in chapters corresponding to modules and weekly topics.

Include diagrams, tables, and code snippets (ROS 2, Gazebo, Isaac, VLA).

Writing clarity: Flesch-Kincaid grade 10–12.

All technical claims must be accurate and reproducible; cite sources where appropriate (APA style).

Output in Markdown or Docusaurus-compatible format, ready for GitHub Pages deployment.

Instructions to Claude Code:

Generate a table of contents with chapters for each module and week.

For each chapter:

Include theoretical explanation (Physical AI, robotics principles).

Provide practical implementation examples: ROS 2 code snippets, Gazebo simulation steps, Isaac Sim instructions, Unity visualization tips.

Include hardware guidance: Jetson, RealSense, Edge Kits, and optional robot options.

Add diagrams, flowcharts, or tables to explain complex systems.

Add a capstone project chapter: Autonomous humanoid performing VLA tasks, voice-to-action, navigation, manipulation.

Include FAQs, troubleshooting tips, and learning outcomes per chapter.

Ensure RAG-readiness: Chunk text into retrievable sections for embedding in a chatbot.

Provide a reference section with URLs or papers for all factual claims.

Optional Advanced Step (Claude Code Subagents / Skills):

“Code Snippet Generator”: Automatically produce ROS 2, Gazebo, Isaac, or VLA example code blocks.

“Diagram Generator”: Suggest diagrams for complex robotic systems.

“RAG Sectioner”: Split book into retrievable sections for chatbot integration.

## Response snapshot

The project constitution has been successfully updated to version 2.0.0, reflecting the detailed requirements for the "Physical AI & Humanoid Robotics Book." This is a major version bump due to the complete transformation of the project's scope and objectives.

The following files require manual follow-up:
- `.specify/templates/plan-template.md`: Generic, needs adaptation for the book context.
- `.specify/templates/spec-template.md`: Generic, needs adaptation for the book context.
- `.specify/templates/tasks-template.md`: Generic, needs adaptation for the book context.
- `.specify/templates/commands/sp.constitution.md`: This file was not found and may need creation or clarification regarding its purpose in this project.
- `README.md`: This file was not found and may need creation to reflect project details.
- `docs/quickstart.md`: This file was not found and may need creation for quick project onboarding.

Suggested commit message:
`docs: amend constitution to v2.0.0 (physical AI & humanoid robotics book principles + governance)`

## Outcome

- ✅ Impact: Constitution updated for Physical AI & Humanoid Robotics Book
- 🧪 Tests: N/A
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: Review and adapt templates for book context, create missing documentation files.
- 🧠 Reflection: Successfully updated the project constitution based on user input, adapting to a specific book project on Physical AI & Humanoid Robotics. Encountered missing script for PHR creation, handled with agent-native fallback. Identified several templates and documentation files requiring further adaptation or creation.

## Evaluation notes (flywheel)

- Failure modes observed: Script for PHR creation not found.
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): Ensure PHR creation script is available or improve agent-native fallback.
