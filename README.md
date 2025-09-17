# WebApp for Wind Turbine Blade Segmentation/Tracking & Camera Parameter Calculations

This is a Flask application for processing videos of wind turbines using YOLO models for blade segmentation and tracking.

## Setup

1. Clone the repository.
2. Create a `config.py` file from the `config.template.py`.
3. Update the model paths in `config.py` to point to your local `.pt` weight files.
4. Install dependencies: `pip install flask ultralytics opencv-python numpy`
5. Run the app: `python app.py`
