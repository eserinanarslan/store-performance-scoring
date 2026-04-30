from flask import Flask, request, jsonify
import warnings

# Suppress non-critical warnings for cleaner logs
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Global objects to store data and model
GLOBAL_DF = None
GLOBAL_MODEL = None
GLOBAL_FEATURES = None


def set_data(df, model, features):
    """
    Store dataset, trained model, and feature list globally
    so they can be accessed inside API endpoints.
    """
    global GLOBAL_DF, GLOBAL_MODEL, GLOBAL_FEATURES
    GLOBAL_DF = df
    GLOBAL_MODEL = model
    GLOBAL_FEATURES = features


def create_app():
    """Return the Flask application instance"""
    return app


@app.route("/health", methods=["GET"])
def health():
    """Simple health check endpoint"""
    return {"status": "ok"}


# =========================
# STORE SUMMARY
# =========================
@app.route("/store/<int:store_id>", methods=["GET"])
def get_store(store_id):
    """
    Return summary statistics for a given store:
    - average score
    - performance label
    - average actual vs expected sales
    - latest trend value
    """
    df = GLOBAL_DF

    store_df = df[df["Store"] == store_id]

    if store_df.empty:
        return jsonify({"error": "store not found"}), 404

    return jsonify({
        "store_id": store_id,
        "avg_score": round(store_df["score"].mean(), 2),
        "performance_label": store_df["performance_label"].value_counts().idxmax(),
        "avg_sales": float(store_df["Sales"].mean()),
        "avg_expected_sales": float(store_df["expected_sales"].mean()),
        "latest_trend": float(store_df["rolling_score"].iloc[-1])
    })


# =========================
# DRIVERS
# =========================
@app.route("/store/<int:store_id>/drivers", methods=["GET"])
def drivers(store_id):
    """
    Compare store-level metrics with global averages
    to highlight key drivers of performance differences.
    """
    df = GLOBAL_DF
    store_df = df[df["Store"] == store_id]

    features = ["conversion_rate", "footfall", "staff_hours", "markdown_rate"]

    result = {}

    for f in features:
        result[f] = round(store_df[f].mean() - df[f].mean(), 3)

    return jsonify({
        "store_id": store_id,
        "drivers_diff_vs_global": result
    })


# =========================
# TOP STORES
# =========================
@app.route("/top-stores", methods=["GET"])
def top_stores():
    """
    Return top-performing stores based on average score.
    Optional query param:
    - limit (default: 10)
    """
    limit = int(request.args.get("limit", 10))

    df = GLOBAL_DF.groupby("Store")["score"].mean().sort_values(ascending=False).head(limit)

    return jsonify(df.reset_index().to_dict(orient="records"))


# =========================
# WORST STORES
# =========================
@app.route("/worst-stores", methods=["GET"])
def worst_stores():
    """
    Return lowest-performing stores based on average score.
    Optional query param:
    - limit (default: 10)
    """
    limit = int(request.args.get("limit", 10))

    df = GLOBAL_DF.groupby("Store")["score"].mean().sort_values().head(limit)

    return jsonify(df.reset_index().to_dict(orient="records"))


# =========================
# SCORE PREDICTION
# =========================
@app.route("/score", methods=["POST"])
def score():
    """
    Predict expected sales for a given input feature set.
    Expects JSON body with model features.
    """
    data = request.json

    try:
        # Prepare input in correct feature order
        X = [[data[f] for f in GLOBAL_FEATURES]]

        # Generate prediction
        pred = GLOBAL_MODEL.predict(X)[0]

        return jsonify({
            "expected_sales": float(pred)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400