# AR-Filter: Real-Time Computer Vision Playground
## 🎬 Demo

Watch the AR filter in action — real-time augmented reality, camera effects, and panorama stitching:
[![AR-Filter Demo](demo.gif)]

> **Turn your webcam into a lab for computer vision magic — no PhD required.**

AR-Filter is a powerful, interactive Python application that brings advanced OpenCV techniques to your fingertips in real time. From augmented reality and panorama stitching to camera calibration and 3D model projection, this tool transforms your laptop into a full-fledged vision research station.

Used by students, hobbyists, and professionals worldwide to learn, prototype, and demo computer vision concepts — all with zero setup beyond `pip install`.

🎯 **One script. Infinite possibilities.**

---

## ✨ What You Can Do?

This isn’t just another filter app. It’s a complete **real-time vision framework** with production-grade algorithms:

- 🔍 **Augmented Reality**: Detect ArUco markers and overlay a 3D T-Rex model in real time.
- 📸 **Panorama Stitching**: Capture frames and generate wide-angle panoramas using feature matching and homography.
- 🎛️ **Camera Calibration**: Calibrate your webcam using a printed chessboard — get accurate intrinsic parameters.
- 🖼️ **Real-Time Filters**: Gaussian blur, bilateral filtering, edge detection, Hough lines, HSV conversion.
- 📊 **Histogram View**: See the live grayscale intensity distribution.
- 🔄 **Geometric Transformations**: Rotate in 90° steps, scale smoothly, or translate manually.
- 🛠️ **Interactive Controls**: Every parameter adjustable via keyboard — no sliders, no GUI overhead.

All running at webcam speed, built on OpenCV and NumPy.

---

## 🚀 Why This Will Go Viral

✅ **It Just Works**  
Clone → Install → Run. No config, no fuss.

✅ **Looks Like Sci-Fi**  
When you project a 3D dinosaur onto a printed marker, people *lean in*. Colleagues stop by. Students ask questions.

✅ **Perfect for Content Creators**  
Ideal for YouTube tutorials, TikTok demos, Twitter/X tech threads, or LinkedIn posts about AI/ML/CV.

✅ **Teaches Without Trying**  
You’ll learn homography, pose estimation, lens distortion, and image pyramids — just by playing.

✅ **Hackable Foundation**  
Add face filters, object tracking, or export videos — it’s open, clean, and well-structured.

✅ **GitHub Gold**  
Has everything trending repos have: visual impact, educational value, and instant gratification.

---

## 💻 Try It in 60 Seconds

### Step 1: Install
```bash
git clone https://github.com/mastersubhajit/AR-Filter.git
cd AR-Filter
```
### Step 2: Set Up Environment
```bash
python -m venv venv
```
#### Activate it
##### On Windows:
```bash
venv\Scripts\activate
```
##### On macOS/Linux:
```bash
source venv/bin/activate
```
##### Now install requirements
```bash
pip install -r requirements.txt
```

## Project Structure
The directory layout is organized for clarity and ease of development:

| Path | Description |
|------|-------------|
| `app.py` | Main entry point. Handles live camera input, real-time effects, AR rendering, user controls, and mode switching. |
| `assets/` | Folder for static assets such as 3D models. |
| `assets/trex_model.obj` | 3D T-Rex model in Wavefront OBJ format used for augmented reality projection. Can be replaced with other models. |
| `calibration.npz` | Auto-generated binary file storing camera calibration data (intrinsic matrix and distortion coefficients). Created after running calibration mode (`[k]`). |
| `requirements.txt` | Lists all Python dependencies. Install via `pip install -r requirements.txt`. |
| `.devcontainer/` | Configuration for VS Code Remote Containers (development environment isolation). |
| `.devcontainer/Dockerfile` | Defines the container image with OpenCV, NumPy, and `opencv-contrib-python`. |
| `.devcontainer/docker-compose.yml` | Optional: supports multi-container setups for advanced development workflows. |
| `README.md` | This document — setup guide, usage instructions, and feature overview. |

## 📋 Keyboard Controls (Table)

Below is a summary of all available keyboard commands:

| Key | Action | Mode | Description |
|-----|--------|------|-------------|
| `0` | Raw video feed | All | Displays unprocessed camera feed |
| `1` | Grayscale mode | All | Converts image to grayscale |
| `2` | HSV color space | All | Switches to HSV color representation |
| `g` | Gaussian blur | Filter | Enables blur with adjustable kernel |
| `f` | Bilateral filter | Filter | Edge-preserving smoothing |
| `e` | Canny edge detection | Edge | Shows detected edges in real time |
| `l` | Hough line detection | Edge | Detects and draws straight lines |
| `r` | Rotate 90° | Transform | Cycles rotation: 0° → 90° → 180° → 270° |
| `[` | Zoom In | Transform | Gradually increases zoom level |
| `]` | Zoom Out | Transform | Gradually decreases zoom level |
| `c` | Increase Contrast | Adjust | Boosts contrast by 0.1 increments |
| `v` | Decrease Contrast | Adjust | Reduces contrast by 0.1 increments |
| `b` | Increase Brightness | Adjust | Brightens image (+5) |
| `n` | Decrease Brightness | Adjust | Darkens image (-5) |
| `o` | Capture Frame | Panorama | Saves current frame for stitching |
| `p` | Stitch Panorama | Panorama | Aligns and blends captured frames |
| `k` | Start Calibration | Calibration | Begin chessboard calibration process |
| `u` | Undistort View | AR/Calibration | Applies lens correction if calibrated |
| `a` | Augmented Reality | AR | Overlays 3D model on ArUco markers |
| `h` | Histogram View | Analysis | Shows real-time intensity distribution |
| `z` | Reset All Settings | All | Resets all parameters to defaults |
| `?` | Toggle Help | UI | Shows or hides sidebar help panel |
| `q` | Quit | All | Destroys the window |


**Note: For resources like Chessboard image, ArUco Marker, please visit assets folder**
