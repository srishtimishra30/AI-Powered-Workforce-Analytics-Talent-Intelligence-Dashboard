# Workforce Insights Dashboard

A full-stack analytics platform for HR and workforce planning teams. It combines a PostgreSQL-backed analytics API, machine learning models for attrition and skill-gap prediction, and a Retrieval-Augmented Generation (RAG) chat assistant grounded in internal HR policy documents — all surfaced through an interactive web dashboard.

**Live demo:**
- 🖥️ Frontend: [creation-of-workforce-insights-dashboard-q5io.onrender.com](https://creation-of-workforce-insights-dashboard-q5io.onrender.com/)
- ⚙️ Backend API docs (Swagger): [creation-of-workforce-insights-dashboard-4cmu.onrender.com/docs](https://creation-of-workforce-insights-dashboard-4cmu.onrender.com/docs)

> Note: Render free-tier services spin down when idle, so the first request after a period of inactivity may take up to a minute to respond.

---

## Features

- **Executive dashboard** — headcount, attrition rate, and department-level breakdowns at a glance
- **Employee directory** — browse and drill into individual employee profiles, roles, and risk indicators
- **Attrition prediction** — ML model estimates the probability that an employee will leave
- **Skill-gap prediction** — ML model flags employees with a projected skill gap
- **At-risk employee list** — surfaces employees ranked by HR red-flag count and burnout risk score
- **Recommendations engine** — generates suggested actions per employee (e.g. training, workload adjustment)
- **HR policy chat assistant** — a hybrid-retrieval RAG pipeline (semantic + BM25 search, reciprocal rank fusion, cross-encoder reranking) answers natural-language questions against an internal HR knowledge base

---

## Architecture

```
┌─────────────────┐        ┌──────────────────────┐        ┌────────────────────┐
│   Frontend       │  HTTP  │   Backend API        │        │   PostgreSQL        │
│   (FastAPI +     │ ─────► │   (FastAPI)          │ ─────► │   employees /       │
│   Jinja2 templates)      │   /analytics          │        │   employee_metrics /│
│   Dashboard, Employees,  │   /predictions        │        │   predictions        │
│   Attrition, Skill Gap,  │   /chat               │        └────────────────────┘
│   Recommendations, Chat  │                       │
└─────────────────┘        │  ┌─────────────────┐  │        ┌────────────────────┐
                            │  │ ML Models        │  │        │ RAG Pipeline        │
                            │  │ (attrition_model,│  │        │ Hybrid retrieval →   │
                            │  │  skill_gap_model)│◄─┼────────│ RRF → reranker →     │
                            │  └─────────────────┘  │        │ context expander →   │
                            │                       │        │ LLM (Groq / Gemini)  │
                            └───────────────────────┘        └────────────────────┘
```

The **Frontend** service also proxies certain calls (`/api/...`) to the **Backend** service via the `BACKEND_URL` environment variable, so both can be deployed independently (as they are on Render).

---

## Tech Stack

| Layer                | Technology                                              |
|-----------------------|----------------------------------------------------------|
| Frontend               | FastAPI, Jinja2, HTML/CSS/JS (static assets)              |
| Backend API            | FastAPI, SQLAlchemy, pg8000                               |
| Database               | PostgreSQL                                                |
| ML / Modeling          | scikit-learn, LightGBM, pandas, numpy, joblib             |
| RAG / Retrieval        | LangChain, FAISS, rank_bm25, cross-encoder reranker       |
| LLM Providers          | Groq, Google Generative AI (Gemini)                       |
| Deployment             | Render (frontend + backend as separate web services)     |

---

## Project Structure

```
.
├── Frontend/                  # FastAPI frontend service (dashboard UI)
│   ├── app.py                 # Routes, templating, chart data prep
│   ├── templates/             # Jinja2 HTML pages
│   └── static/                # CSS/JS/assets
├── backend/                    # FastAPI backend API service
│   ├── main.py                 # App entrypoint, router registration
│   ├── database.py             # DB session/engine setup
│   └── routers/
│       ├── analytics.py        # /analytics/summary, /by-department, /at-risk
│       ├── predictions.py      # /predictions/attrition, /predictions/skill-gap
│       └── chat.py             # /chat — RAG-powered Q&A
├── Machine Learning/
│   ├── train_attrition.py      # Trains the attrition classifier
│   ├── train_skill_gap.py      # Trains the skill-gap model
│   ├── models/                 # Saved model artifacts (.pkl)
│   └── Featured Engineering.csv# Engineered feature dataset
├── RAG/
│   ├── main.py                  # End-to-end RAG pipeline orchestration
│   ├── hybrid_retrieve.py       # Semantic + BM25 hybrid retrieval, RRF fusion
│   ├── reranker.py              # Cross-encoder reranking
│   ├── context_expander.py      # Expands retrieved chunks with context
│   ├── chunk_documents.py       # Document chunking
│   ├── create_vectorstore.py    # Builds the FAISS vector store
│   ├── llm.py                   # LLM call wrapper (Groq/Gemini)
│   ├── knowledge_base/          # Source HR policy documents
│   └── vectorstore/              # Persisted FAISS index
├── database/
│   ├── schema.sql               # Postgres table definitions
│   └── load_to_postgres.py      # Loads processed data into Postgres
├── recommendation/
│   └── recommendation_engine.py # Generates per-employee recommendations
├── data/                        # Raw and processed datasets
├── outputs/                     # Prediction + recommendation outputs
├── notebook/
│   └── EDA1.ipynb               # Exploratory data analysis
├── requirements.txt
├── start_backend.ps1            # Loads .env and starts the backend (Windows)
├── test_all_endpoints.py        # Backend endpoint smoke tests
└── test_frontend_pages.py       # Frontend page smoke tests
```

---

## API Reference

Full interactive documentation is available at `/docs` on the backend service (Swagger UI) — see the [live docs](https://creation-of-workforce-insights-dashboard-4cmu.onrender.com/docs).

### Analytics
| Method | Endpoint                     | Description                                                |
|--------|-------------------------------|--------------------------------------------------------------|
| GET    | `/analytics/summary`          | Total employees, attrition count, and attrition rate         |
| GET    | `/analytics/by-department`    | Headcount and attrition rate grouped by department            |
| GET    | `/analytics/at-risk?limit=20` | Employees ranked by HR red-flag count and burnout risk score  |

### Predictions
| Method | Endpoint                    | Description                                          |
|--------|------------------------------|--------------------------------------------------------|
| POST   | `/predictions/attrition`     | Returns an attrition probability + predicted label      |
| POST   | `/predictions/skill-gap`     | Returns a skill-gap prediction                           |

Example request body for both:
```json
{
  "age": 34,
  "monthly_income": 5200,
  "years_at_company": 4,
  "overall_satisfaction_index": 0.62,
  "burnout_risk_score": 0.41,
  "absence_rate_per_year": 0.05,
  "is_new_hire": 0,
  "overtime_and_low_satisfaction_flag": 0
}
```

### Chat (RAG assistant)
| Method | Endpoint | Description                                                    |
|--------|----------|--------------------------------------------------------------------|
| POST   | `/chat`  | Ask a natural-language HR/workforce policy question              |

```json
{ "message": "What is the policy for remote work?" }
```

### Health
| Method | Endpoint  | Description             |
|--------|-----------|---------------------------|
| GET    | `/`       | Service status message    |
| GET    | `/health` | Basic health check          |

---

## Getting Started (Local Development)

### Prerequisites
- Python 3.10+
- PostgreSQL instance
- API keys for Groq and/or Google Generative AI (for the chat assistant)

### 1. Clone and install dependencies
```bash
git clone https://github.com/srishtimishra30/Creation-of-Workforce-Insights-Dashboard-for-Employee-Skill-and-Analytics.git
cd Creation-of-Workforce-Insights-Dashboard-for-Employee-Skill-and-Analytics
pip install -r requirements.txt
```

### 2. Configure environment variables
Create a `.env` file in the project root:
```env
DATABASE_URL=postgresql+pg8000://<user>:<password>@<host>:<port>/<database>
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
BACKEND_URL=http://127.0.0.1:8000
```

### 3. Set up the database
```bash
psql -U <user> -d <database> -f database/schema.sql
python database/load_to_postgres.py
```

### 4. Train the ML models (optional — pretrained artifacts are included in `Machine Learning/models/`)
```bash
python "Machine Learning/train_attrition.py"
python "Machine Learning/train_skill_gap.py"
```

### 5. Build the RAG vector store (optional — a prebuilt index is included in `RAG/vectorstore/`)
```bash
python RAG/create_vectorstore.py
```

### 6. Run the backend API
```bash
# Windows
./start_backend.ps1

# macOS/Linux
uvicorn backend.main:app --reload --port 8000
```
API will be available at `http://127.0.0.1:8000`, docs at `http://127.0.0.1:8000/docs`.

### 7. Run the frontend
```bash
uvicorn Frontend.app:app --reload --port 5000
```
Dashboard will be available at `http://127.0.0.1:5000`.

---

## Testing

```bash
python test_all_endpoints.py      # Backend endpoint smoke tests
python test_frontend_pages.py     # Frontend page smoke tests
```

---

## Deployment

Both services are deployed independently on [Render](https://render.com):

- **Backend** — a FastAPI web service exposing the `/analytics`, `/predictions`, and `/chat` routes, plus interactive docs at `/docs`.
- **Frontend** — a separate FastAPI + Jinja2 web service that renders the dashboard UI and proxies API calls to the backend via the `BACKEND_URL` environment variable.

To deploy your own instance:
1. Deploy `backend/` as a web service with start command `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`, and set `DATABASE_URL`, `GROQ_API_KEY`, `GOOGLE_API_KEY` as environment variables.
2. Deploy `Frontend/` as a separate web service with start command `uvicorn Frontend.app:app --host 0.0.0.0 --port $PORT`, and set `BACKEND_URL` to the backend's deployed URL.

---

## License

This project is licensed under the [MIT License](LICENSE).
