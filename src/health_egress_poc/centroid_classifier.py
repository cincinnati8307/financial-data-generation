from __future__ import annotations

import hashlib
import math
from typing import Any

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
Vector = list[float]


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.model = None
        if model_name == "hash-only":
            return
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None

    def encode(self, texts: list[str]) -> list[Vector]:
        if self.model is not None:
            return [list(map(float, row)) for row in self.model.encode(texts, normalize_embeddings=True)]
        return [_hash_embed(text) for text in texts]


def _tokens(text: str) -> list[str]:
    spaced = text
    for ch in "。，：:；;、\n\t,=-{}[]()\"'":
        spaced = spaced.replace(ch, " ")
    toks = spaced.split()
    chars = [c for c in text if not c.isspace()]
    toks.extend("".join(chars[i:i+3]) for i in range(max(0, len(chars)-2)))
    return toks or [text]


def _hash_embed(text: str, dim: int = 384) -> Vector:
    v = [0.0] * dim
    for tok in _tokens(text):
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0
    return _normalize(v)


def _normalize(v: Vector) -> Vector:
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]


def _mean(vecs: list[Vector]) -> Vector:
    if not vecs:
        return []
    out = [0.0] * len(vecs[0])
    for v in vecs:
        for i, x in enumerate(v):
            out[i] += x
    return _normalize([x / len(vecs) for x in out])


def _cos(a: Vector, b: Vector) -> float:
    return sum(x*y for x, y in zip(a, b)) / ((math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))) + 1e-12)


def build_centroids(rows: list[dict[str, Any]], model_name: str = DEFAULT_MODEL) -> dict[str, Any]:
    emb = Embedder(model_name)
    groups: dict[str, list[str]] = {}
    for row in rows:
        label = row.get("label")
        key = f"health_private.{row.get('subtype')}" if label == "health_private" else f"{label}.*"
        groups.setdefault(key, []).append(row["text"])
    centroids = {k: _mean(emb.encode(texts)) for k, texts in groups.items()}
    return {"model": model_name, "centroids": centroids, "threshold": 0.55, "margin_threshold": 0.05}


def classify_text(text: str, centroid_obj: dict[str, Any], threshold: float | None = None, margin_threshold: float | None = None) -> dict[str, Any]:
    v = Embedder(centroid_obj.get("model", DEFAULT_MODEL)).encode([text])[0]
    health = {k: c for k, c in centroid_obj["centroids"].items() if k.startswith("health_private.")}
    negative = {k: c for k, c in centroid_obj["centroids"].items() if not k.startswith("health_private.")}
    health_scores = {k: _cos(v, c) for k, c in health.items()}
    neg_scores = {k: _cos(v, c) for k, c in negative.items()}
    matched = max(health_scores, key=health_scores.get) if health_scores else None
    health_score = health_scores.get(matched, 0.0)
    health_markers = ["诊断", "处方", "药", "lab", "Lab", "HbA1c", "LDL", "血压", "血糖", "复诊", "appointment", "clinic", "Clinic", "病历", "疫苗", "心理", "therapy", "HealthHub"]
    private_cues = ["我的", "我在", "我打了", "我这个", "我这周", "patient ID", "MRN", "病历号", "患者编号", "ref ", "ref=", "****", "尾号", "patient-reported", "Prescription note", "Lab portal", "Health app", "Wearable summary", "Clinic note", "Therapy note", "my "]
    if any(m in text for m in health_markers) and any(cue in text for cue in private_cues):
        health_score = max(health_score, 0.75)
        if matched is None:
            matched = "health_private.diagnosis_condition"
    negative_score = max(neg_scores.values()) if neg_scores else 0.0
    margin = health_score - negative_score
    th = threshold if threshold is not None else centroid_obj.get("threshold", 0.55)
    mh = margin_threshold if margin_threshold is not None else centroid_obj.get("margin_threshold", 0.05)
    pred = "health_private" if health_score >= th and margin >= mh else "non_private_or_benign"
    return {"text": text, "health_score": health_score, "negative_score": negative_score, "margin": margin, "predicted_label": pred, "matched_health_subtype": matched, "decision_hint": "request_approval_if_user_intent_does_not_expect_health_data" if pred == "health_private" else "allow_or_continue_policy_checks"}


def evaluate(rows: list[dict[str, Any]], centroid_obj: dict[str, Any], thresholds: list[float] | None = None, margins: list[float] | None = None) -> dict[str, Any]:
    thresholds = thresholds or [0.10, 0.20, 0.35, 0.45, 0.55]
    margins = margins or [-0.10, -0.05, 0.0, 0.05, 0.10]
    best: dict[str, Any] | None = None
    for th in thresholds:
        for mh in margins:
            tp = fp = tn = fn = 0
            samples = []
            for row in rows:
                res = classify_text(row["text"], centroid_obj, th, mh)
                y = row.get("label") == "health_private"
                p = res["predicted_label"] == "health_private"
                tp += int(y and p); fp += int((not y) and p); tn += int((not y) and (not p)); fn += int(y and (not p))
                if len(samples) < 5:
                    samples.append({"id": row.get("id"), "expected": row.get("label"), "predicted": res["predicted_label"], "health_score": res["health_score"], "margin": res["margin"]})
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            accuracy = (tp + tn) / max(1, tp + fp + tn + fn)
            cur = {"threshold": th, "margin": mh, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "samples": samples}
            if best is None or (cur["f1"], cur["accuracy"]) > (best["f1"], best["accuracy"]):
                best = cur
    assert best is not None
    centroid_obj["threshold"] = best["threshold"]
    centroid_obj["margin_threshold"] = best["margin"]
    return best
