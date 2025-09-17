import os
import time
from flask import (
    Blueprint, request, redirect, url_for, send_from_directory,
    render_template, Response, current_app
)
from werkzeug.utils import secure_filename

# Import our project-specific modules
from .processing import process_video, gen_webcam_frames
from .models import yolo_v8, yolo_v11n, yolo_v11s, active_model_lock

# 'main' is the name of our blueprint
main = Blueprint('main', __name__)

@main.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            return redirect(url_for("main.index"))

        filename = secure_filename(f.filename)
        in_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        f.save(in_path)

        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        out_name = f"processed_{ts}_{filename}"
        out_path = os.path.join(current_app.config['OUTPUT_FOLDER'], out_name)

        try:
            process_video(in_path, out_path)
            return render_template("index.html", video_url=url_for("main.serve_output", filename=out_name))
        except Exception as e:
            print(f"Error processing video: {e}")
            return "Error processing video.", 500

    return render_template("index.html", video_url=None)


@main.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filename)


@main.route("/webcam_feed")
def webcam_feed():
    return Response(gen_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@main.route("/set_model/<model_name>")
def set_model(model_name):
    # This is a bit tricky with module-level globals, but works for development servers.
    # We need to modify the global 'active_model' in the 'models' module.
    from . import models

    with active_model_lock:
        if model_name.lower() == "v8":
            models.active_model = models.yolo_v8
        elif model_name.lower() == "v11n":
            models.active_model = models.yolo_v11n
        elif model_name.lower() == "v11s":
            models.active_model = models.yolo_v11s
        else:
            return f"Unknown model {model_name}", 400
    return f"Switched to model {model_name.upper()}"