from flask import Flask, jsonify
from flask_cors import CORS

from backend.routes.lsrw_routes import lsrw_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(lsrw_bp)


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "success",
        "message": "Adaptive LSRW GD API is running"
    })


if __name__ == "__main__":
    app.run(debug=True)