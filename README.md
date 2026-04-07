---
title: Personalized HCM Assistant
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Personalized HCM Benefits Assistant

An AI-powered benefits chatbot that provides personalized guidance to employees using RAG over a benefits knowledge corpus, combined with per-employee persistent memory.

## Features

- **Personalized responses** — each employee's profile, enrolled benefits, and conversation history inform every answer
- **Proactive recommendations** — relevant benefits surfaced on login based on profile gaps
- **Bidirectional memory** — post-chat state update adds new interests/questions and removes resolved ones
- **State compaction** — keeps employee state within a 1500-token budget across sessions
- **Persistent state** — stored in Supabase Postgres (survives restarts)

## Stack

- Gradio 4.x UI
- OpenAI GPT-4o + text-embedding-3-small
- ChromaDB (in-process vector store)
- Supabase Postgres (employee state persistence)

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon/service-role key |
