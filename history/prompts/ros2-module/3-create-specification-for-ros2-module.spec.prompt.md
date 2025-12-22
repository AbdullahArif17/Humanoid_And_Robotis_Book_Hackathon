---
id: 3
title: Create Specification for ROS 2 Module
stage: spec
date: 2025-12-05
surface: agent
model: claude-sonnet-4-5-20250929
feature: ros2-module
branch: 1-ros2-module
user: user
command: /sp.specify You are an expert in Physical AI and Humanoid Robotics. Generate a detailed module specification for: **Module 1: The Robotic Nervous System (ROS 2)** Include the following sections in **specify.md** format: 1. **Module Overview** - Focus and theme - Key concepts (ROS 2 Nodes, Topics, Services, URDF, rclpy integration) - Connection to Physical AI and humanoid robotics 2. **Learning Objectives** - What students will be able to achieve after completing this module 3. **Prerequisites** - Software, hardware, and knowledge requirements 4. **Tasks / Labs / Exercises** - Step-by-step practical exercises for building ROS 2 packages - Example code snippets in Python (rclpy) - How to simulate nodes and services 5. **Assessment Criteria** - How student work will be evaluated - Expected outputs, success criteria 6. **RAG-ready Sections** - Chunk module content into retrievable pieces suitable for embedding into a chatbot 7. **References & Resourc
labels: ["spec", "ros2-module"]
links:
  spec: specs/1-ros2-module/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/1-ros2-module/spec.md
  - specs/1-ros2-module/checklists/requirements.md
tests: null
---

## Prompt

/sp.specify You are an expert in Physical AI and Humanoid Robotics. Generate a detailed module specification for: **Module 1: The Robotic Nervous System (ROS 2)** Include the following sections in **specify.md** format: 1. **Module Overview** - Focus and theme - Key concepts (ROS 2 Nodes, Topics, Services, URDF, rclpy integration) - Connection to Physical AI and humanoid robotics 2. **Learning Objectives** - What students will be able to achieve after completing this module 3. **Prerequisites** - Software, hardware, and knowledge requirements 4. **Tasks / Labs / Exercises** - Step-by-step practical exercises for building ROS 2 packages - Example code snippets in Python (rclpy) - How to simulate nodes and services 5. **Assessment Criteria** - How student work will be evaluated - Expected outputs, success criteria 6. **RAG-ready Sections** - Chunk module content into retrievable pieces suitable for embedding into a chatbot 7. **References & Resourc

## Response snapshot

New branch `1-ros2-module` created.
Specification for "ROS 2 - The Robotic Nervous System" created at `specs/1-ros2-module/spec.md`.
Quality checklist created at `specs/1-ros2-module/checklists/requirements.md`.
The specification is ready for planning (`/sp.plan`).

## Outcome

- ✅ Impact: Specification for ROS 2 module created.
- 🧪 Tests: N/A
- 📁 Files: Created specs/1-ros2-module/spec.md and specs/1-ros2-module/checklists/requirements.md
- 🔁 Next prompts: Proceed to planning using /sp.plan.
- 🧠 Reflection: Successfully created the spec and checklist for the ROS 2 module. Handled script failure by manually creating branch and files, then using agent-native PHR creation.

## Evaluation notes (flywheel)

- Failure modes observed: Script for PHR creation not found; PowerShell not available.
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): Investigate pwsh availability and update scripts or documentation for manual fallback.
