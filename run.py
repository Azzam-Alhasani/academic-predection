"""
Entry point for the Academic Performance Monitoring & Prediction System.
Run this file to start the Flask server (local development).

Safe defaults:
  - Binds to 127.0.0.1 (localhost) so student data is not exposed on the network.
  - Debug mode is controlled by the selected configuration (development = on).
"""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    host = os.environ.get("HOST", app.config.get("HOST", "127.0.0.1"))
    port = int(os.environ.get("PORT", app.config.get("PORT", 5000)))
    debug = app.config.get("DEBUG", False)
    app.run(debug=debug, host=host, port=port)