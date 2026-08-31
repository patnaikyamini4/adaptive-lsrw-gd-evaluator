from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "success",
        "message": "Adaptive LSRW GD API is running"
    })


if __name__ == "__main__":
    app.run(debug=True)