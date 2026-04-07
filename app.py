# ---------------------------------------------------------------------------
# Monkey-patch gradio_client bool-schema bug bundled with gradio==4.44.1.
# The same fix is applied via setup.sh locally; here we patch at runtime so
# HF Spaces (where we can't edit installed packages) works without changes.
# Must happen BEFORE `import gradio`.
# ---------------------------------------------------------------------------
import gradio_client.utils as _gc_utils

# Patch get_type: handle bool schemas (e.g. additionalProperties: true/false)
_orig_get_type = _gc_utils.get_type


def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "any"
    return _orig_get_type(schema)


_gc_utils.get_type = _patched_get_type

# Patch _json_schema_to_python_type: also guard at the recursive entry point
# so bool schemas passed as additionalProperties/items never reach get_type unchecked.
_orig_j2p = _gc_utils._json_schema_to_python_type


def _patched_j2p(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_j2p(schema, defs)


_gc_utils._json_schema_to_python_type = _patched_j2p
# ---------------------------------------------------------------------------

import gradio as gr
import json
from pathlib import Path

from src import vector_store, rag, state_manager, recommender, memory_updater, state_compactor

_CORPUS_DIR = Path(__file__).parent / "corpus"

# Initialise ChromaDB collection at startup (loads corpus + cached embeddings)
print("[App] Initialising vector store…")
vector_store.init_vector_store()
print("[App] Vector store ready.")

# Load available employee names from JSON files
_EMPLOYEE_NAMES = state_manager.get_employee_names()

# Load corpus topics for the quick-access buttons
_CORPUS_TOPICS = vector_store.get_corpus_topics()


def format_state_display(state):
    if not state:
        return "No employee selected."
    display = {k: v for k, v in state.items() if k != "employee_id"}
    return json.dumps(display, indent=2, default=str)


def on_employee_select(employee_name, state):
    if not employee_name or employee_name == "":
        return [], "No employee selected.", {}

    state = state_manager.load(employee_name)
    state["session_counter"] += 1

    # Compaction runs on login, before recommender, to keep state within token budget
    try:
        state = state_compactor.compact(state)
    except Exception as exc:
        print(f"[App] State compaction error: {exc}")

    try:
        greeting = recommender.generate_greeting(state)
    except Exception as exc:
        greeting = f"Hi {state['name']}! Welcome to Acme Corp Benefits Assistant. How can I help you today?"
        print(f"[Recommender] Failed to generate greeting: {exc}")
    chatbot_history = [{"role": "assistant", "content": greeting}]

    return chatbot_history, format_state_display(state), state


def on_chat_message(user_message, history, state):
    if not user_message or not user_message.strip():
        return history, "", state

    if not state:
        history = history + [{"role": "assistant", "content": "Please select an employee first."}]
        return history, "", state

    history = history + [{"role": "user", "content": user_message}]

    # Pass history without the current turn (rag.answer appends it internally)
    prior_history = history[:-1]
    reply = rag.answer(user_message, prior_history, employee_state=state)
    history = history + [{"role": "assistant", "content": reply}]

    return history, "", state


def on_end_chat(history, state):
    if not state:
        gr.Info("No active session to save.")
        return history, "No employee selected.", state

    name = state.get("name", "Employee")

    try:
        state = memory_updater.update_state(history, state)
        gr.Info(f"Memory updated and session saved for {name}!")
    except Exception as exc:
        print(f"[App] Memory update error: {exc}")
        gr.Warning(f"State saved for {name} (memory update skipped due to an error).")

    state_manager.save(state)

    return history, format_state_display(state), state


# --- Gradio UI ---

with gr.Blocks(
    title="Acme Corp Benefits Assistant",
    theme=gr.themes.Soft(),
    css="""
        footer { display: none !important; }
        .topic-btn { font-size: 0.82rem !important; }
        .sidebar-panel {
            position: sticky !important;
            top: 1rem;
        }
    """,
) as app:
    employee_state = gr.State(value={})

    gr.Markdown("# 🏢 Acme Corp Benefits Assistant")
    gr.Markdown("Personalized benefits guidance powered by AI. Select your employee profile to begin.")

    with gr.Accordion("📖 User Guide", open=False):
        gr.Markdown("""
## How to use this app

This is a prototype **personalized benefits chatbot** for Acme Corp employees.
It demonstrates how an AI assistant can give context-aware benefits guidance
based on each employee's profile, enrollment status, and conversation history.

---

### Getting started

1. **Select an employee** from the dropdown at the top-left.
   - The assistant will greet you with a proactive recommendation tailored to that employee's situation.
   - Three synthetic personas are available: Maria Chen, James Williams, and Priya Sharma — each with different tenure, family status, and enrollment.

2. **Ask a question** by typing in the message box and pressing **Send** (or Enter).
   - Example questions: *"What is the 401k match?"*, *"Can I use my HSA for dental expenses?"*, *"How much PTO do I accrue?"*
   - Or click any **Browse Topics** button on the right to pre-fill a starter question.

3. **End the session** by clicking **End Chat & Save**.
   - The assistant analyzes the conversation and updates the employee's memory:
     saves new interests, records unresolved questions, marks resolved ones as done, and logs action items.
   - The **What I Know About You** panel (right sidebar) refreshes to show the updated state.

---

### The three employee personas

| Employee | Age | Situation | Key interests |
|---|---|---|---|
| **Maria Chen** | 28 | New hire (3 months), single | 401k match, tuition reimbursement |
| **James Williams** | 42 | 8 years, married + 2 kids | FSA vs HSA, tax-advantaged accounts |
| **Priya Sharma** | 55 | 15 years, married, VP Finance | Catch-up contributions, Medicare transition |

---

### What the sidebar panels do

- **Browse Topics** — quick-access buttons that pre-fill a relevant starter question for any benefits area.
- **What I Know About You** — shows the employee's live state JSON (interests, unresolved questions, pending actions, interaction history). This is for prototype evaluation only — it would not be visible to employees in production.

---

### Tips

- Try selecting the **same employee across multiple sessions** to see memory build up over time.
- Ask about a topic, end the chat, reselect the same employee — notice their interests and history have been updated.
- The assistant will **not repeat recommendations** it has already surfaced (tracked via `seen_recommendations`).
""")

    with gr.Row():
        with gr.Column(scale=3):
            employee_dropdown = gr.Dropdown(
                choices=[""] + _EMPLOYEE_NAMES,
                value="",
                label="Select Employee",
                info="Simulates employee login",
            )

            chatbot = gr.Chatbot(
                label="Chat",
                height=450,
                type="messages",
                show_copy_button=True,
                render_markdown=True,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Ask about your benefits...",
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            end_chat_btn = gr.Button("End Chat & Save", variant="stop")

        with gr.Column(scale=2, elem_classes=["sidebar-panel"]):
            # --- Topic quick-access buttons (dynamic from corpus) ---
            gr.Markdown("### 📚 Browse Topics\n*Click a topic to pre-fill a starter question.*")
            _topic_buttons = []
            for t in _CORPUS_TOPICS:
                _topic_buttons.append(
                    gr.Button(t["label"], size="sm", elem_classes=["topic-btn"])
                )

            gr.Markdown("---")
            gr.Markdown("### 🧠 What I Know About You\n*For prototype evaluation only — not visible to employees in production.*")
            state_display = gr.Textbox(
                label="Employee State (JSON)",
                value="No employee selected.",
                lines=20,
                max_lines=30,
                interactive=False,
            )

    # --- Event wiring ---

    # Wire each topic button to pre-fill the message input
    for _btn, _topic in zip(_topic_buttons, _CORPUS_TOPICS):
        _question = _topic["question"]
        _btn.click(
            fn=lambda q=_question: q,
            inputs=[],
            outputs=[msg_input],
        )

    employee_dropdown.change(
        fn=on_employee_select,
        inputs=[employee_dropdown, employee_state],
        outputs=[chatbot, state_display, employee_state],
    )

    msg_input.submit(
        fn=on_chat_message,
        inputs=[msg_input, chatbot, employee_state],
        outputs=[chatbot, msg_input, employee_state],
    )

    send_btn.click(
        fn=on_chat_message,
        inputs=[msg_input, chatbot, employee_state],
        outputs=[chatbot, msg_input, employee_state],
    )

    end_chat_btn.click(
        fn=on_end_chat,
        inputs=[chatbot, employee_state],
        outputs=[chatbot, state_display, employee_state],
    )


if __name__ == "__main__":
    # HF Spaces exposes port 7860 on 0.0.0.0; do NOT pass server_name/server_port
    # when running there — Spaces sets SPACE_ID env var.
    import os as _os
    if _os.environ.get("SPACE_ID"):
        app.launch(allowed_paths=[str(_CORPUS_DIR)])
    else:
        app.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=[str(_CORPUS_DIR)])
