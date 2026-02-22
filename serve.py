"""Serving entry point. Starts the FastAPI server via uvicorn."""

import uvicorn

from serving.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
