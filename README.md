# Vehicle Speed Estimation using YOLOv8 and ByteTrack

A computer vision project for vehicle detection, tracking, counting, segmentation, and approximate speed estimation from traffic videos.

## Features

- Vehicle detection using YOLOv8
- Multi-object tracking using ByteTrack
- Track ID preservation across frames
- Vehicle segmentation using YOLOv8-seg
- Approximate speed estimation using centroid displacement
- Lane-width based pixel-to-meter calibration
- Unique truck counting
- Annotated video output
- CSV output with tracking and speed information

## Technologies

- Python
- YOLOv8
- ByteTrack
- OpenCV
- NumPy
- Pandas

## How It Works

The system detects and tracks vehicles in a traffic video using YOLOv8 and ByteTrack.

Speed is estimated approximately every 15 frames using the displacement of the vehicle centroid. A calibration factor based on lane width is used to convert pixel movement into real-world distance.

The current version focuses on trucks and generates both an annotated output video and a CSV file containing tracking and speed information.

## Output

The CSV output includes information such as:

- Frame number
- Track ID
- Vehicle class
- Detection confidence
- Centroid coordinates
- Pixel displacement
- Estimated speed in km/h
- Unique truck count

## Files

- `main.py` — Main detection, tracking, segmentation, and speed estimation code
- `speeds_sample.csv` — Sample output results
## Sample Output Video

Sample output video file:
[output_speed.mp4](output_speed.mp4)

## Installation

Install the required Python packages:

```bash
pip install ultralytics opencv-python pandas numpy
