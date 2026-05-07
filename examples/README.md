# Examples

Two end-to-end examples that exercise JeevesAgent's own loader, vector
store, retriever-as-tool pattern, and multi-agent architectures —
nothing pulled in from outside the framework.

| File | What it shows |
|---|---|
| [`01_rag_pdf.py`](01_rag_pdf.py) | Single-agent RAG over a folder of PDFs. Loader → `RecursiveChunker` → `ChromaVectorStore` → `@tool` retriever → `Agent`. |
| [`02_specialist_debate.py`](02_specialist_debate.py) | Five domain specialists (IT / physics / medicine / finance / law), each with their own folder of PDFs and their own Chroma collection, composed via `Team.debate(...)` with a synthesising judge agent. |

Both examples generate small sample PDFs on first run (via
`reportlab`) and cache them under `examples/data/`. The on-disk
Chroma indices are also cached, so subsequent runs only re-execute
the agent loop against OpenAI.

## Run

```bash
# .env should contain OPENAI_API_KEY=sk-...
python examples/01_rag_pdf.py
python examples/02_specialist_debate.py
```

## What's wired up

```
01_rag_pdf.py
─────────────
  examples/data/general/
    company_handbook.pdf
    engineering_guide.pdf
    security_policy.pdf
    support_runbook.pdf
        │
        ▼  jeevesagent.loader.load(...)
    Document(content=<markdown>)
        │
        ▼  RecursiveChunker(chunk_size=600).split(...)
    list[Chunk]
        │
        ▼  ChromaVectorStore.add(chunks)   (persisted on disk)
    indexed collection 'general_docs'
        │
        ▼  @tool search_docs(query): wraps store.search(query, k=4)
    Agent(model="gpt-4.1-mini", tools=[search_docs])

02_specialist_debate.py
───────────────────────
  examples/data/it/         examples/data/physics/    ...
    it_runbook.pdf            physics_notes.pdf       ...
        │                         │
        ▼                         ▼
  Chroma 'it_docs'         Chroma 'physics_docs'      ...
        │                         │
        ▼                         ▼
  search_it_docs           search_physics_docs        ...
        │                         │
        ▼                         ▼
  Agent (IT tech)          Agent (Physicist)         ...

  Team.debate(
    debaters=[it, phys, med, fin, law],
    judge=Agent("...synthesis judge..."),
    rounds=1,
  )
```
