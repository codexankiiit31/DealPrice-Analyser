# 🔌 Internal API Integration

DealSense AI relies on internal API calls to trigger remote AI agents hosted on **Modal** and **OpenAI**. These are not public endpoints but are securely invoked within the application backend.

---

## 🧠 OpenAI API

OpenAI is used by two agents:

- `DealScannerAgent`: selects the top 5 deals from raw inputs, ensuring clear descriptions and explicit prices.
- `RAGPriceAgent`: estimates product prices after retrieving similar items from ChromaDB to provide contextual grounding.

---

## 🔐 Authentication

Access to external services requires the following environment variables, which should be defined in the `.env` file.

- `OPENAI_API_KEY`
- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

!!! note
    All API interactions are internal — there is no public REST or HTTP API exposed.
