# ChurnInsight — Customer Churn Prediction (FastAPI)

A responsive browser UI backed by a FastAPI REST API and an existing scikit-learn customer-churn pipeline. The UI collects customer details and shows the model's churn/stay prediction, probability estimate and risk band.

## Project files

```text
customer-churn-fastapi/
├── app.py                 # FastAPI endpoints and model inference
├── index.html             # Front-end form and result display
├── churn_pipeline.pkl     # Required trained model (copy from your model repo)
├── requirements.txt
├── .python-version        # Python version for Render
├── render.yaml            # Render deployment settings
└── examples/
    └── sample_request.json
```

## Required model file

This package intentionally does not replace or rename the trained model. Place `churn_pipeline.pkl` in this same folder before running or deploying. The model is from the existing public repository: https://github.com/sumbalzaheer75/customer-prediction

Keep the scikit-learn version aligned with the version used when the pipeline was saved. This project pins scikit-learn to `1.7.2`.

## Run locally on Windows

Activate the virtual environment in PowerShell, then install packages:

```powershell
python -m pip install -r requirements.txt
```

Start the service:

```powershell
python -m uvicorn app:app --reload
```

Open:

- UI: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
- Model health check: http://127.0.0.1:8000/health

## Deploy to Render

1. Create a new GitHub repository named `customer-churn-fastapi`.
2. Upload all project files to the repository, including `churn_pipeline.pkl`. Do not upload `.venv/` or any personal/customer data.
3. Go to https://render.com/ and connect the GitHub repository.
4. Create a **Web Service**. If Render asks for settings, use:
   - Runtime: Python 3
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - Plan: Free (if available in your account/region)
5. Wait for the deployment to finish. Open the `onrender.com` URL. The custom UI is at `/`, docs at `/docs`, and health at `/health`.

The included `render.yaml` sets these values for Render Blueprint-based deployment as well.

## Endpoints

- `GET /` — responsive web UI
- `GET /health` — confirms the model file is present
- `POST /predict` — validates a customer JSON payload and returns prediction, probability, risk band, and message
- `GET /docs` — interactive OpenAPI documentation

## Example request

Use `examples/sample_request.json` from the `/docs` page's `POST /predict` endpoint.

## Portfolio description

**ChurnInsight — Customer Churn Prediction | FastAPI | Python | Scikit-learn**

Built a responsive web interface around a trained customer churn pipeline. Served predictions through FastAPI, validated customer inputs with Pydantic, returned probability-based risk bands, and documented endpoints using OpenAPI/Swagger.

> For educational demonstration only. Predictions are model estimates, not certainty about customer behaviour.
