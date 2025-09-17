import os
from flask import Flask

def create_app():
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    # Load configuration from config.py
    app.config.from_pyfile('../config.py')

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Ensure upload and output folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

    # Register the blueprint from routes.py
    from . import routes
    app.register_blueprint(routes.main)

    return app