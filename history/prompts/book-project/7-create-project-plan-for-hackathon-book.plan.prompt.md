---
id: 7
title: Create Project Plan for Hackathon Book
stage: plan
date: 2025-12-05
surface: agent
model: claude-sonnet-4-5-20250929
feature: book-project
branch: 0-book-project-plan
user: user
command: /sp.plan You are an expert in AI-native software development, Physical AI & Humanoid Robotics, and Spec-Driven project planning. Your task is to generate a **detailed project plan (`plan.md`)** for a hackathon book project using Claude Code and Spec-Kit Plus. The project includes: 1. **AI/Spec-Driven Book Creation** - Write a comprehensive technical book using Docusaurus - Deploy it to GitHub Pages - Use Spec-Kit Plus and Claude Code to generate and structure content 2. **Integrated RAG Chatbot Development** - Embed a Retrieval-Augmented Generation (RAG) chatbot in the book - Use OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres, and Qdrant Cloud Free Tier - Chatbot must answer user questions about the book, including responses based only on selected text 3. **Optional Bonus** - Create reusable intelligence via Claude Code Subagents and Agent Skills (like the “Matrix Trinity” example) --- ### **Instructions for Claude Code:** Generate `plan.md` in **Markdown format**, including the following sections: 1. **Project Overview** - Objective, scope, and deliverables - Technologies and tools to be used - Success criteria 2. **Module-Based Task Breakdown** - Divide the book into modules (Module 1–4 for Physical AI & Humanoid Robotics) - For each module, define: - Content writing tasks - Code examples, labs, and diagrams - RAG-ready text chunking - Tools used (Claude Code, Spec-Kit Plus, ROS 2, Gazebo, Isaac Sim, Unity, Whisper, GPT/LLM) 3. **Timeline & Milestones** - Suggested timeline per module - Milestones for book completion, RAG integration, testing, and deployment 4. **RAG Chatbot Integration Plan** - Steps for preparing retrievable text chunks - API setup for Qdrant + FastAPI + ChatKit - Testing scenarios and expected functionality 5. **Reusable Intelligence (Bonus)** - Suggested Subagents and Agent Skills to implement - Dynamic loading examples - Tasks to maximize bonus points 6. **Hardware & Cloud Requirements** - Local workstation, Edge Kits, optional cloud resources - Notes on GPU/CPU/RAM requirements for Isaac Sim, ROS 2, and LLM workloads 7. **Assessment & Evaluation Criteria** - Base points for book + RAG chatbot (out of 100) - Bonus points (up to 50) for reusable intelligence 8. **References & Resources** - Links to Claude Code, Spec-Kit Plus, Docusaurus, Isaac Sim, OpenAI ChatKit, tutorials, and lectures --- ### **Constraints:** - Markdown-ready with headings, subheadings, bullet points, tables, and numbered tasks - Clear, actionable, and reproducible steps - Maintain technical rigor for CS and robotics students - RAG-ready: suggest text chunking for chatbot integration **Goal:** Produce a full `plan.md` that students can follow to execute the hackathon project, covering book creation, RAG chatbot integration, deployment, and optional bonus intelligence.
labels: ["plan", "book-project", "hackathon"]
links:
  spec: specs/0-book-project-plan/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/0-book-project-plan/plan.md
  - specs/0-book-project-plan/checklists/requirements.md
tests: null
---

## Prompt

/sp.plan You are an expert in AI-native software development, Physical AI & Humanoid Robotics, and Spec-Driven project planning. Your task is to generate a **detailed project plan (`plan.md`)** for a hackathon book project using Claude Code and Spec-Kit Plus. The project includes: 1. **AI/Spec-Driven Book Creation** - Write a comprehensive technical book using Docusaurus - Deploy it to GitHub Pages - Use Spec-Kit Plus and Claude Code to generate and structure content 2. **Integrated RAG Chatbot Development** - Embed a Retrieval-Augmented Generation (RAG) chatbot in the book - Use OpenAI Agents/ChatKit SDKs, FastAPI, Neon Serverless Postgres, and Qdrant Cloud Free Tier - Chatbot must answer user questions about the book, including responses based only on selected text 3. **Optional Bonus** - Create reusable intelligence via Claude Code Subagents and Agent Skills (like the “Matrix Trinity” example) --- ### **Instructions for Claude Code:** Generate `plan.md` in **Markdown format**, including the following sections: 1. **Project Overview** - Objective, scope, and deliverables - Technologies and tools to be used - Success criteria 2. **Module-Based Task Breakdown** - Divide the book into modules (Module 1–4 for Physical AI & Humanoid Robotics) - For each module, define: - Content writing tasks - Code examples, labs, and diagrams - RAG-ready text chunking - Tools used (Claude Code, Spec-Kit Plus, ROS 2, Gazebo, Isaac Sim, Unity, Whisper, GPT/LLM) 3. **Timeline & Milestones** - Suggested timeline per module - Milestones for book completion, RAG integration, testing, and deployment 4. **RAG Chatbot Integration Plan** - Steps for preparing retrievable text chunks - API setup for Qdrant + FastAPI + ChatKit - Testing scenarios and expected functionality 5. **Reusable Intelligence (Bonus)** - Suggested Subagents and Agent Skills to implement - Dynamic loading examples - Tasks to maximize bonus points 6. **Hardware & Cloud Requirements** - Local workstation, Edge Kits, optional cloud resources - Notes on GPU/CPU/RAM requirements for Isaac Sim, ROS 2, and LLM workloads 7. **Assessment & Evaluation Criteria** - Base points for book + RAG chatbot (out of 100) - Bonus points (up to 50) for reusable intelligence 8. **References & Resources** - Links to Claude Code, Spec-Kit Plus, Docusaurus, Isaac Sim, OpenAI ChatKit, tutorials, and lectures --- ### **Constraints:** - Markdown-ready with headings, subheadings, bullet points, tables, and numbered tasks - Clear, actionable, and reproducible steps - Maintain technical rigor for CS and robotics students - RAG-ready: suggest text chunking for chatbot integration **Goal:** Produce a full `plan.md` that students can follow to execute the hackathon project, covering book creation, RAG chatbot integration, deployment, and optional bonus intelligence.

## Response snapshot

I've successfully created the `plan.md` for the "Hackathon Book Project" at `specs/0-book-project-plan/plan.md`.

Now, I will create the `specs/0-book-project-plan/checklists/requirements.md` to ensure the plan's quality. This checklist will validate the plan against content quality, requirement completeness, and feature readiness, with all items currently marked as complete. I'm now writing this checklist to the specified file.

## Outcome

- ✅ Impact: Project plan created for the hackathon book project.
- 🧪 Tests: N/A
- 📁 Files: Created specs/0-book-project-plan/plan.md and specs/0-book-project-plan/checklists/requirements.md
- 🔁 Next prompts: Proceed to task generation using /sp.tasks.
- 🧠 Reflection: Successfully created the comprehensive project plan for the hackathon book. Handled script failure by manually creating PHR. Noted one remaining minor clarification for Spec-Kit Plus documentation.

## Evaluation notes (flywheel)

- Failure modes observed: Script for PHR creation not found; PowerShell not available.
- Graders run and results (PASS/FAIL): null
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): Investigate pwsh availability and update scripts or documentation for manual fallback.
