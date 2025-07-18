# 📘 Vibe Coding Guidebook: Best Practices for Clean & Lean Code

---

## 🎯 Core Philosophy

> ✨ *"Write less, do more. Prioritize clarity. Reuse intelligently. Execute with intention."*

---

## 🔹 1. Lean Code, Always

* 🔁 **Reuse** existing functions/scripts wherever possible
* ❌ Avoid rewriting or duplicating logic unless **absolutely necessary**
* 🧼 Keep functions small and focused on one task
* 🧠 Think: *“Can I solve this with what already exists?”*

---

## 🔹 2. Keep It Simple & Readable

* 👁️ Prefer clarity over cleverness
* do not edit /modify any existing legacy scripts 
* always read docs folder for context setting and instructions
* always stick to the data struture mentioned in the docs folder for data processing and analysis and function handling 
* ☝️ Stick to consistent naming conventions (snake\_case for functions/vars)
* 🧱 Use docstrings and inline comments only where necessary—not excessive
* ✅ Use descriptive variable/function names (`credit_utilization` > `cu`)

---

## 🔹 3. Modular by Default

* 🧩 Break down complex processes into reusable modules
* ❌ Avoid monolithic scripts—organize logic across clearly named files
* 📂 Maintain a clean directory structure (`/utils`, `/output`, `/models`, etc.)

---

## 🔹 4. Respect the Base

* ⚖️ If a core script exists (e.g., `feature_engineering.py`), use it
* �� Don’t overwrite or fork the base unless:

  * a major refactor is **justified**, and
  * legacy compatibility is handled

---

## 🔹 5. Fail Gracefully

* ❗ Use try–except blocks for critical logic paths
* 🧵 Log failures with context: `logger.error(f"Missing credit data for agent {agent_id}")`

---

## 🔹 6. Minimal Dependencies

* 📦 Avoid unnecessary imports or libraries
* 🔧 Use built-in Python features unless a library gives a **clear advantage**
* 💡 Prefer `pandas`, `numpy`, `json`, `datetime`, `logging`—only add beyond that if justified

---

## 🔹 7. Consistent Output Patterns

* 📁 Output must be saved in structured, predictable paths:

  * `/output/processed/` – Intermediate and final data
  * `/output/logs/` – Logs
  * `/output/email_summaries/` – Markdown mailers
* 📜 Use JSON or clean CSVs for traceable outputs

---

## 🔹 8. Interactive, Not Intrusive

* 🧱 Use CLI menus only after processing (not during core logic)
* ⌨️ Prompt clearly:

  ```python
  print("1. Agent Lookup\n2. Region Mailer\nq. Quit\n> ", end="")
  ```

---

## 🔹 9. Respect Performance

* 🐍 Use vectorized operations (e.g., `pandas.apply` only when necessary)
* ⏱️ Avoid loops over dataframes unless no alternative
* 🫶 Cache heavy calculations wherever repeatable

---

## 🔹 10. Version Control Aware

* 🗃️ Keep functions and configs loosely coupled
* 📁 Maintain a `config.py` or `.env` for changeable parameters (paths, flags)
* 🧪 Keep experimentation separate from production (`/experiments/` folder if needed)

---

## ✅ TL;DR — *Golden Rules for LLMs*

```
1. Don’t duplicate – reuse modules unless change is justified.
2. Keep code modular, lean, and traceable.
3. Structure outputs and logs consistently.
4. Write for humans — clear, readable, intentional code.
5. Default to simplicity. Scale with structure, not shortcuts.
```

---
