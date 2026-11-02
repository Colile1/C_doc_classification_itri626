"""Results dashboard: serves ui/static, a read-only JSON API and a live classifier.

Shows the prepared dataset, the three-family comparison and each model's
results, all read from the committed split and metric files, and classifies an
uploaded page with the best checkpoint present on the machine. The pages need
only the standard library; the classifier needs the project environment.
Run from the repository root:

    python ui/server.py

then open http://127.0.0.1:8626.

Author: Colile Sibanda
"""
import csv
import json
import os
import sys
import threading
import warnings
from email import policy
from email.parser import BytesParser
from io import BytesIO
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import pathlib
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
PORT = 8626

_predictor = None
_predict_lock = threading.Lock()
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"}


def split_class_counts():
    """Read the prepared split summary, grouped by document class."""
    path = REPO / "data" / "splits" / "split_summary.csv"
    counts = {}
    if not path.exists():
        return counts
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts.setdefault(row["class_name"], {})[row["split"]] = int(row["count"])
    return counts


def extraction_status():
    """Per-class page counts in each split, from the committed split summary."""
    counts = split_class_counts()
    names = sorted(counts)
    rows = []
    for name in names:
        split = counts[name]
        prepared = sum(split.values())
        rows.append({
            "class": name,
            "labelled": prepared,
            "train": split.get("train", 0),
            "validation": split.get("val", 0),
            "test": split.get("test", 0),
            "extracted_pct": 100.0 if prepared else 0.0,
            "in_data_raw": prepared,
        })
    prepared_total = sum(row["labelled"] for row in rows)
    split_ready = bool(rows) and all(row["labelled"] for row in rows)
    return {
        "extracted_dir": "data/splits",
        "labels_present": bool(counts),
        "estimated_pct": 100.0 if split_ready else 0.0,
        "prepared_total": prepared_total,
        "classes": rows,
    }


def training_runs():
    """Summarize every recorded training run, newest family first."""
    runs = []
    for entry in MODEL_CATALOGUE:
        name, base = entry["name"], REPO / entry["dir"]
        history = _csv_rows(base / "metrics" / f"history_{name}.csv")
        metrics = _csv_rows(base / "metrics" / f"metrics_{name}.csv")
        if not history and not metrics:
            continue
        best = None
        if history:
            try:
                best = max(history, key=lambda row: float(row["val_macro_f1"]))
            except (KeyError, ValueError):
                best = None
        seconds = sum(float(row.get("seconds") or 0) for row in history)
        runs.append({
            "model": name,
            "label": entry["label"],
            "family": entry["family"],
            "role": entry["role"],
            "platform": entry["platform"],
            "config": entry.get("config", f"configs/{name}.yaml"),
            "epochs": len(history),
            "best_epoch": int(best["epoch"]) if best else None,
            "best_val_macro_f1": float(best["val_macro_f1"]) if best else None,
            "test_accuracy": float(metrics[0]["accuracy"]) if metrics else None,
            "test_macro_f1": float(metrics[0]["macro_f1"]) if metrics else None,
            "seconds": round(seconds, 1),
            "evaluated": bool(metrics),
        })
    return {"runs": runs, "count": len(runs)}


def results():
    """Return the three-family comparison written by run_compare.py."""
    path = REPO / "outputs" / "comparison" / "comparison.csv"
    rows = _csv_rows(path)
    return {"exists": bool(rows), "rows": rows, "source": "comparison" if rows else None}


def pulse():
    """Change signature for the live-refresh poller: the comparison file's age."""
    path = REPO / "outputs" / "comparison" / "comparison.csv"
    counts = extraction_status()
    return {
        "exists": path.exists(),
        "running": False,
        "mtime": int(path.stat().st_mtime) if path.exists() else 0,
        "total": counts["prepared_total"],
    }


# The three models in the three-family comparison, with where each run writes
# and an assessment drawn from its recorded metrics.
MODEL_CATALOGUE = [
    {
        "name": "baseline_cnn_v4",
        "label": "Scratch CNN (v4)",
        "family": "CNN from scratch",
        "dir": "runs/baseline_cnn_v4",
        "role": "comparison",
        "platform": "Colab Tesla T4, batch 32",
        "summary": (
            "A small convolutional network trained from random initialisation. "
            "It is the reference the pretrained models must beat to justify "
            "their cost."
        ),
        "pros": [
            "Smallest by far: 1.18 M parameters, 4.5 MB on disk",
            "Fastest inference at 1.5 ms per page",
            "No pretrained weights, so no external licence or download",
            "Already competitive on the visually distinctive classes",
            "Wins no class outright here, but is within 0.03 F1 on resume, "
            "advertisement and handwritten",
        ],
        "cons": [
            "Lowest overall score: macro-F1 0.8308",
            "Weakest on the confusable business documents, where it loses "
            "0.08 to 0.11 F1 against the pretrained models",
            "Form is its hardest class at F1 0.7129",
            "Learns only from the 11 200 project pages",
        ],
    },
    {
        "name": "resnet50",
        "label": "ResNet50 (pretrained CNN)",
        "family": "Pretrained convolutional",
        "dir": "runs/resnet50",
        "role": "comparison",
        "platform": "Colab Tesla T4, batch 32",
        "summary": (
            "ImageNet weights with conv1, bn1, layer1 and layer2 frozen, so "
            "22.1 M of 23.5 M parameters were fine-tuned on document pages."
        ),
        "pros": [
            "Within 0.009 macro-F1 of the best model",
            "4.1x faster per page than ViT-B/16 (3.3 ms against 13.4 ms)",
            "3.6x smaller than ViT-B/16 (90 MB against 327 MB)",
            "Wins four of the eight classes outright",
            "The best accuracy-to-cost balance of the three",
        ],
        "cons": [
            "Still 20x larger than the from-scratch baseline",
            "Overfits: training loss fell to 0.017 against a validation loss "
            "near 0.62",
            "Loses to ViT-B/16 on the three hardest classes",
            "Freezing choice needs justifying; it was not tuned",
        ],
    },
    {
        "name": "vit_b16",
        "label": "ViT-B/16 (pretrained transformer)",
        "family": "Pretrained transformer",
        "dir": "runs/vit_b16",
        "role": "comparison",
        "platform": "Colab Tesla T4, batch 32",
        "summary": (
            "A vision transformer with ImageNet weights, fine-tuned end to end. "
            "Self-attention relates distant parts of a page rather than pooling "
            "local patches."
        ),
        "pros": [
            "Best overall: accuracy 0.8829, macro-F1 0.8828",
            "Wins the three hardest classes: form, budget and invoice",
            "Largest gain over the baseline on invoice, +0.106 F1",
            "Validation macro-F1 was still improving at the final epoch",
        ],
        "cons": [
            "Largest and slowest: 85.8 M parameters, 327 MB, 13.4 ms per page",
            "Leads ResNet50 by only 0.009 macro-F1 for 4x the inference cost",
            "Overfits hardest: training loss 0.0074 against validation 0.58",
            "73x the parameters of the baseline for 0.052 macro-F1",
        ],
    },
]


def _csv_rows(path):
    """Rows of a CSV as dictionaries, or an empty list when it is absent."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def models():
    """Every run with its metrics, per-class scores, history and assessment."""
    catalogue = []
    for entry in MODEL_CATALOGUE:
        name, base = entry["name"], REPO / entry["dir"]
        metrics = _csv_rows(base / "metrics" / f"metrics_{name}.csv")
        per_class = _csv_rows(base / "metrics" / f"per_class_{name}.csv")
        history = _csv_rows(base / "metrics" / f"history_{name}.csv")
        efficiency = _csv_rows(base / "metrics" / f"efficiency_{name}.csv")
        errors = _csv_rows(base / "metrics" / f"errors_{name}.csv")

        figures = []
        figure_dir = base / "figures"
        if figure_dir.is_dir():
            figures = sorted(p.name for p in figure_dir.glob("*.png"))

        best = None
        if history:
            try:
                best = max(history, key=lambda row: float(row["val_macro_f1"]))
            except (KeyError, ValueError):
                best = None

        catalogue.append({
            **{k: entry[k] for k in
               ("name", "label", "family", "role", "platform", "summary", "pros", "cons")},
            "dir": entry["dir"],
            "evaluated": bool(metrics),
            "metrics": metrics[0] if metrics else None,
            "per_class": per_class,
            "efficiency": efficiency[0] if efficiency else None,
            "epochs_run": len(history),
            "best_epoch": int(best["epoch"]) if best else None,
            "best_val_macro_f1": float(best["val_macro_f1"]) if best else None,
            "history": history,
            "top_errors": errors[:8],
            "figures": figures,
        })

    comparison = _csv_rows(REPO / "outputs" / "comparison" / "comparison.csv")
    strengths = _csv_rows(REPO / "outputs" / "comparison" / "class_strengths.csv")
    report_figures = []
    report_dir = REPO / "outputs" / "report_figures"
    if report_dir.is_dir():
        report_figures = sorted(p.name for p in report_dir.glob("*.png"))

    families = {row["family"] for row in comparison if row.get("family")}
    return {
        "models": catalogue,
        "comparison": comparison,
        "class_strengths": strengths,
        "report_figures": report_figures,
        "families_complete": len(families),
        "families_required": 3,
    }

ROUTES = {
    "/api/extraction": extraction_status,
    "/api/results": results,
    "/api/models": models,
    "/api/training-runs": training_runs,
    "/api/pulse": pulse,
}


def predict_document(image_bytes, filename, content_type):
    """Classify one uploaded page with the best checkpoint available locally."""
    global _predictor
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError("Upload a JPG, PNG, TIFF, BMP, or WebP document image.")
    if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("The upload is empty or larger than 15 MB.")

    # The dashboard can be started from a lightweight system Python while the
    # model dependencies live in the project's external virtual environment.
    package_dirs = []
    env_venv = os.environ.get("ITRI626_VENV_DIR")
    if env_venv:
        package_dirs.extend(Path(env_venv).glob("lib/python*/site-packages"))
    package_dirs.extend(Path("/tmp/itri626-venv/lib").glob("python*/site-packages"))
    package_dirs.extend((Path.home() / ".venvs/itri626/lib").glob("python*/site-packages"))
    for package_dir in package_dirs:
        if package_dir.is_dir() and str(package_dir) not in sys.path:
            sys.path.insert(0, str(package_dir))
    source_dir = str(REPO / "src")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    try:
        from PIL import Image, UnidentifiedImageError
        import torch
        from docclf.config.loader import load_config
        from docclf.data.transforms import build_transforms
        from docclf.models.registry import build_model
    except ImportError as error:
        raise RuntimeError(
            "Classifier dependencies are unavailable. Activate the project environment "
            "or set ITRI626_VENV_DIR to its directory."
        ) from error

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image_bytes)) as image:
                if image.width * image.height > 40_000_000:
                    raise ValueError("The image is too large to process (maximum 40 megapixels).")
                image.verify()
            with Image.open(BytesIO(image_bytes)) as image:
                page = image.convert("L")
                page.load()
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("This file is not a readable document image.") from error

    # Classify with the best model whose weights are actually on this machine,
    # in descending order of held-out macro-F1. Checkpoints are large and are
    # not committed, so whichever ones were copied back decide what is served.
    candidates = [
        ("vit_b16", "configs/vit_b16.yaml", "runs/vit_b16/checkpoints/vit_b16_best.pt"),
        ("resnet50", "configs/resnet50.yaml", "runs/resnet50/checkpoints/resnet50_best.pt"),
        ("baseline_cnn_v4", "configs/baseline_cnn_v4.yaml",
         "runs/baseline_cnn_v4/checkpoints/baseline_cnn_v4_best.pt"),
    ]
    chosen = next(
        ((name, REPO / config, REPO / checkpoint)
         for name, config, checkpoint in candidates
         if (REPO / checkpoint).is_file() and (REPO / config).is_file()),
        None,
    )
    if chosen is None:
        raise RuntimeError(
            "No trained checkpoint is present. Copy one model's "
            "<model>_best.pt into its runs/<model>/checkpoints/ directory, or "
            "train a model first."
        )
    model_name, config_path, checkpoint_path = chosen
    with _predict_lock:
        if _predictor is None:
            cfg = load_config(config_path)
            model = build_model(model_name, cfg.n_classes, cfg)
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            _predictor = (
                model, build_transforms(cfg, train=False), tuple(cfg.classes),
                int(payload.get("epoch", 0)), model_name,
            )
        model, transform, classes, checkpoint_epoch, served_model = _predictor
        tensor = transform(page).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor)[0], dim=0).cpu().tolist()
    ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    return {
        "filename": Path(filename or "uploaded image").name,
        "predicted_class": ranked[0][0],
        "confidence": round(ranked[0][1], 6),
        "scores": [{"class": name, "probability": round(score, 6)} for name, score in ranked],
        "model": served_model,
        "checkpoint_epoch": checkpoint_epoch,
        "note": "Scores are model probabilities and may not be calibrated confidence estimates.",
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        # Figures live under outputs/, not in static/, so they are served from
        # there by an explicit whitelist: /figure/<area>/<name>.png only, with
        # no path separators, so nothing outside those folders can be read.
        if path.startswith("/figure/"):
            parts = path[len("/figure/"):].split("/")
            areas = {
                "report": REPO / "outputs" / "report_figures",
                "comparison": REPO / "outputs" / "comparison" / "figures",
            }
            for model in ("baseline_cnn_v4", "resnet50", "vit_b16"):
                areas[model] = REPO / "runs" / model / "figures"
            if len(parts) != 2 or parts[0] not in areas or not parts[1].endswith(".png"):
                self.send_error(404)
                return
            figure = areas[parts[0]] / pathlib.PurePosixPath(parts[1]).name
            if not figure.is_file():
                self.send_error(404)
                return
            body = figure.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        route = ROUTES.get(path)
        if route is None:
            return super().do_GET()
        try:
            body, code = json.dumps(route()).encode(), 200
        except Exception as err:  # the page shows this text, so keep it specific
            body, code = json.dumps({"detail": f"{type(err).__name__}: {err}"}).encode(), 500
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD_BYTES + 1024 * 1024:
                raise ValueError("Choose a document image smaller than 15 MB.")
            content_type = self.headers.get("Content-Type", "")
            if not content_type.lower().startswith("multipart/form-data"):
                raise ValueError("Send the document as a file upload.")
            message = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
                + self.rfile.read(length)
            )
            upload = None
            for part in message.iter_parts():
                if part.get_param("name", header="content-disposition") == "document":
                    upload = (part.get_filename(), part.get_content_type(), part.get_payload(decode=True))
                    break
            if not upload:
                raise ValueError("Choose a document image to classify.")
            filename, file_type, payload = upload
            response = predict_document(payload or b"", filename, file_type)
            body, code = json.dumps(response).encode(), 200
        except ValueError as error:
            body, code = json.dumps({"detail": str(error)}).encode(), 400
        except Exception as error:
            body, code = json.dumps({"detail": f"{type(error).__name__}: {error}"}).encode(), 500
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # keep the console quiet; the pulse polls every few seconds


if __name__ == "__main__":
    print(f"Dashboard on http://127.0.0.1:{PORT}  (Ctrl+C to stop)")
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
