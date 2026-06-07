# NephroVax RAG Prototype

A retrieval-augmented generation (RAG) prototype for the NephroVax vaccination guideline,
covering pneumococcal, meningococcal, and Hib vaccines for patients on complement
inhibitor therapy.

> **This is a research prototype, not a clinical tool. Do not use for patient care.**

## What's in here

```
nephrovax-rag/
├── app.py                      # Streamlit UI
├── nephrovax/                  # RAG library (importable, testable)
│   ├── config.py
│   ├── loader.py
│   ├── chunker.py
│   ├── retrieval.py
│   └── generation.py
├── guidelines/                 # Source markdown files
├── evals/                      # Retrieval test cases
├── requirements.txt
└── .streamlit/                 # Streamlit config + secrets template
```

## Running locally

```bash
# 1. Install
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your API keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your real keys

# 3. Run
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploying to Streamlit Community Cloud

1. **Push this repo to GitHub** (private is fine; recommended given `confidential: true` on the guidelines).
2. **Go to https://share.streamlit.io** and sign in with GitHub.
3. **"New app"** → select your repo and branch → set "Main file path" to `app.py`.
4. **Click "Advanced settings" → "Secrets"** and paste:
   ```toml
   VOYAGE_API_KEY = "your-real-key"
   ANTHROPIC_API_KEY = "your-real-key"
   # Optional: APP_PASSCODE = "choose-a-passcode"
   ```
5. **Deploy.** First boot takes 2-3 minutes (installing dependencies). Subsequent boots are faster.

The app sleeps after inactivity; first visit after sleep takes 30-60 seconds to wake. Tell your users.

## API keys you need

- **Voyage AI** (embeddings): https://www.voyageai.com — free tier covers this easily
- **Anthropic** (generation): https://console.anthropic.com — pay per use, expect ~$0.01-0.03/query

## Architecture notes

- **Embedding model:** `voyage-3` (outperformed `voyage-3-large` on the 24-case evaluation harness — see `evals/`).
- **Chunking:** one chunk per leaf section. Natural-language context header prepended for retrieval.
- **Retrieval:** FAISS flat inner product (exact, fine at this scale).
- **Generation:** Claude with a system prompt that requires citation and forbids invention.

## Updating the guidelines

1. Edit a markdown file in `guidelines/`.
2. Run `python scripts/validate_guidelines.py guidelines/` to check structural compliance.
3. Commit and push. Streamlit Cloud will redeploy automatically.
4. The cached index will rebuild on first request (this is intentional).

## Limitations

- Retrieval brittleness to paraphrase (esp. for "C5 inhibitor" vs "complement inhibitor"-style queries).
- Multi-vaccine queries may not surface all relevant chunks.
- The Hib guidance reflects NephroVax's interpretation, which extends beyond CDC's explicit adult Hib indications.
- 24-case test harness is small; not a substitute for clinical validation.

See `evals/` for the full evaluation methodology.
