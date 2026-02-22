"""
Serving App
===========

FastAPI application that loads GEPA-optimised programs from disk and
classifies text against all six relationship labels.

Start via::

    python serve.py
"""

import json
import os
from contextlib import asynccontextmanager

import dspy
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from serving.config import (
    LABELS,
    LM_BASE_URL,
    LM_MAX_TOKENS,
    LM_REASONING_EFFORT,
    LM_TEMPERATURE,
    POSITIVE_CLASS,
    PROGRAMS_MANIFEST,
    PROGRAMS_SUBDIR,
    RESULTS_DIR,
)

_programs: dict[str, dspy.Module] = {}
_manifest: dict[str, str] = {}


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    labels: list[str]
    details: dict[str, bool]


def _load_programs() -> None:
    """Loads all saved programs and the manifest from disk.

    Args:
        None.

    Returns:
        None.
    """
    programs_dir = os.path.join(RESULTS_DIR, PROGRAMS_SUBDIR)
    manifest_path = os.path.join(programs_dir, PROGRAMS_MANIFEST)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run training first: python train.py"
        )
    with open(manifest_path) as f:
        manifest_data = json.load(f)
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"Invalid manifest format at {manifest_path}. Expected a JSON object."
        )
    missing_labels = [label for label in LABELS if label not in manifest_data]
    if missing_labels:
        raise ValueError(
            f"Manifest at {manifest_path} is missing labels: {missing_labels}. "
            "Re-run training to regenerate complete programs."
        )
    _manifest.clear()
    _manifest.update(manifest_data)
    anthropic_labels = [
        label
        for label in LABELS
        if str(_manifest.get(label, "")).startswith("anthropic/")
    ]
    if anthropic_labels:
        raise ValueError(
            "Anthropic models are not supported in this project. "
            f"Unsupported manifest labels: {anthropic_labels}."
        )
    _programs.clear()
    for label in LABELS:
        label_dir = os.path.join(programs_dir, label)
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(
                f"Program directory not found for '{label}' at {label_dir}."
            )
        try:
            _programs[label] = dspy.load(label_dir, allow_pickle=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load program for '{label}' from {label_dir}: {exc}"
            ) from exc
        logger.info(
            f"[{label}] Program loaded from {label_dir} (gen_model={_manifest[label]})."
        )
    logger.info(f"All {len(_programs)} programs loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook that loads programs at startup.

    Args:
        app: FastAPI application instance (unused directly).

    Returns:
        Async generator yielding control to the running application.
    """
    _load_programs()
    yield


app = FastAPI(title="Relationship Classifier", lifespan=lifespan)


@app.get("/health")
def health():
    """Returns service health and loaded-program count.

    Args:
        None.

    Returns:
        Dict with status and number of loaded label programs.
    """
    return {"status": "ok", "labels_loaded": len(_programs)}


@app.post("/classify", response_model=ClassifyResponse)
def classify(request: ClassifyRequest):
    """Runs inference for all labels on the input text.

    Args:
        request: Incoming request payload with a single text field.

    Returns:
        ClassifyResponse containing positive label names and boolean results per label.
    """
    if not _programs:
        raise HTTPException(status_code=503, detail="Programs not loaded.")
    details = {}
    for label in LABELS:
        try:
            gen_model = _manifest[label]
            lm_kwargs = {"temperature": LM_TEMPERATURE, "max_tokens": LM_MAX_TOKENS}
            if LM_BASE_URL:
                lm_kwargs["api_base"] = LM_BASE_URL
            name = gen_model.split("/")[-1] if "/" in gen_model else gen_model
            if LM_REASONING_EFFORT and name.startswith("o"):
                lm_kwargs["reasoning_effort"] = LM_REASONING_EFFORT
            lm = dspy.LM(gen_model, **lm_kwargs)
            with dspy.context(lm=lm):
                pred = _programs[label](text=request.text)
                pred_label = str(pred.label).strip().lower()
        except Exception as exc:
            logger.exception(f"[{label}] Inference request failed.")
            raise HTTPException(
                status_code=502,
                detail=f"Inference failed for label '{label}'.",
            ) from exc
        details[label] = pred_label == POSITIVE_CLASS
    positive_labels = [label for label, is_pos in details.items() if is_pos]
    return ClassifyResponse(labels=positive_labels, details=details)
