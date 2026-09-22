from flask import Flask, jsonify
from flask_cors import CORS

from backend.services.mongodb import client, db
from backend.routes.lsrw_routes import lsrw_bp


app = Flask(__name__)

CORS(app)

# Register LSRW routes
app.register_blueprint(lsrw_bp)


# --------------------------------------------------
# Root route
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Adaptive LSRW GD API is running",
        "service": "backend"
    })


# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "success",
        "message": "Adaptive LSRW GD API is running"
    })


# --------------------------------------------------
# MongoDB connection test
# --------------------------------------------------
@app.route("/api/database-test", methods=["GET"])
def database_test():
    try:
        client.admin.command("ping")

        return jsonify({
            "status": "success",
            "message": "MongoDB connection is working",
            "database": db.name
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "MongoDB connection failed",
            "error": str(e)
        }), 500


# --------------------------------------------------
# Start Flask application
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)