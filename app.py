"""Flask website for scoring DDI pairs across saved models."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from src.web.model_service import engine


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    drug1 = ""
    drug2 = ""
    abstract = ""
    report = None
    error = None

    if request.method == "POST":
        drug1 = request.form.get("drug1", "").strip()
        drug2 = request.form.get("drug2", "").strip()
        abstract = request.form.get("abstract", "").strip()

        if not drug1 or not drug2:
            error = "Please enter both drug names."
        else:
            report = engine.infer_pair(drug1=drug1, drug2=drug2, abstract=abstract)

    return render_template(
        "index.html",
        drug1=drug1,
        drug2=drug2,
        abstract=abstract,
        report=report,
        results=[] if report is None else report.get("model_outputs", []),
        error=error,
        result_count=0 if report is None else len(report.get("model_outputs", [])),
    )


@app.route("/predict-ddi", methods=["POST"])
def predict_ddi():
    payload = request.get_json(silent=True) or {}
    drug_a = str(payload.get("drug_a", "")).strip()
    drug_b = str(payload.get("drug_b", "")).strip()
    abstract = str(payload.get("abstract", "")).strip()

    if not drug_a or not drug_b:
        return jsonify({"error": "drug_a and drug_b are required"}), 400

    report = engine.infer_pair(drug1=drug_a, drug2=drug_b, abstract=abstract)
    return jsonify(report)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)