import cv2
import numpy as np
import time
import os


def draw_text(frame, text, pos=(20, 50), scale=1.0, color=(255, 255, 255), thickness_override=None):
    """Draws white text with a black outline for better visibility."""
    thickness = thickness_override or max(1, int(scale * 2))
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


class LiveCameraEffects:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        self.window_name = "AR-Filter"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # --- State Variables ---
        self.mode = "raw"
        self.show_help = True  # Toggle help with '?'

        # Basic Adjustments
        self.contrast = 1.0
        self.brightness = 0

        # Geometric Transforms
        self.rotation_angle_index = 0
        self.rotation_angles = [0, 90, 180, 270]
        self.scale_index = 1
        self.scales = [0.5, 1.0, 1.5, 2.0]
        self.current_scale = self.scales[self.scale_index]

        # Filter Parameters
        self.gaussian_kernel = 9
        self.bilateral_params = {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75}

        # Panorama
        self.panorama_frames = []
        self.panorama_result = None

        # Calibration
        self.mtx, self.dist = None, None
        self.calibration_results = {}
        self._load_calibration()
        self.CHESSBOARD_SIZE = (9, 6)
        self.SQUARE_SIZE_MM = 25
        self.objp = np.zeros((self.CHESSBOARD_SIZE[0] * self.CHESSBOARD_SIZE[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.CHESSBOARD_SIZE[0], 0:self.CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * self.SQUARE_SIZE_MM
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objpoints, self.imgpoints = [], []
        self.last_capture_time = time.time()
        self.TARGET_IMAGES = 20

        # AR Setup
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        except AttributeError:
            print("Warning: Legacy ArUco API detected.")
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None

        # 3D Model
        self.model_vertices = []
        self.model_faces = []
        model_path = './assets/trex_model.obj'
        if not os.path.exists(model_path):
            model_path = 'trex_model.obj'
        self._load_obj(model_path, scale_factor=0.0005)

        # Help Menu
        self.help_lines = [
            "[?] Toggle Help   [q] Quit",
            "",
            "[0] Raw    [1] Grayscale    [2] HSV",
            "[a] Augmented Reality",
            "[g] Gaussian Blur    [f] Bilateral Filter",
            "[e] Canny Edge       [l] Hough Lines",
            "[h] Histogram View",
            "",
            "[c/v] Contrast +/-   [b/n] Brightness +/-",
            "",
            "[r] Rotate 90 degrees",
            "[t] Fixed Translate  [s] Cycle Scale Presets",
            "[k] Start Calibration  [u] Undistort View",
            "[o] Capture Frame    [p] Stitch Panorama",
            "[z] Reset All Settings",
            "",
            "--- Controls ---",
            "Scale:     [ and ] to zoom in/out",
            "Gaussian:  +/- to change kernel size",
            "Bilateral: d/D, m/M, x/X for params",
        ]
        print("Ready! Press '?' to toggle help.")

    def _load_calibration(self):
        """Loads intrinsic matrix and distortion coefficients."""
        if os.path.exists('calibration.npz'):
            try:
                data = np.load('calibration.npz')
                self.mtx, self.dist = data['mtx'], data['dist']
                print("Calibration data loaded from 'calibration.npz'.")
            except Exception as e:
                print(f"ERROR: Could not load calibration data: {e}")
        else:
            print("WARNING: 'calibration.npz' not found. AR and undistort require calibration.")

    def _load_obj(self, filename, scale_factor=0.0003):
        """Parses .OBJ file into vertices and face indices."""
        if not os.path.exists(filename):
            print(f"ERROR: Model file '{filename}' not found.")
            return
        vertices, faces = [], []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        parts = line.strip().split()[1:4]
                        if len(parts) == 3:
                            try:
                                x, y, z = map(float, parts)
                                vertices.append([x, y, z])
                            except ValueError:
                                continue
                    elif line.startswith('f '):
                        indices = []
                        for part in line.strip().split()[1:]:
                            idx_str = part.split('/')[0]
                            try:
                                indices.append(int(idx_str) - 1)
                            except (ValueError, IndexError):
                                continue
                        if len(indices) >= 3:
                            faces.append(indices)
            if not vertices:
                print("ERROR: No vertices found in the OBJ file.")
                return
            verts_np = np.array(vertices, dtype=np.float32)
            center = np.mean(verts_np, axis=0)
            self.model_vertices = (verts_np - center) * scale_factor
            self.model_faces = faces
            print(f"Loaded {len(vertices)} vertices and {len(faces)} faces from '{filename}'")
        except Exception as e:
            print(f"Error loading OBJ file {filename}: {e}")

    # --- Improved Panorama Functions ---

    def _calculate_homography_orb(self, img_src, img_dst):
        """Compute homography using ORB features for panorama alignment."""
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(img_src, None)
        kp2, des2 = orb.detectAndCompute(img_dst, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 15:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def _stitch_two_images(self, right_img, left_img):
        """Stitches right image onto the left base image with better blending."""
        H = self._calculate_homography_orb(right_img, left_img)
        if H is None:
            print("Homography could not be computed.")
            return left_img.copy()

        h1, w1 = right_img.shape[:2]
        h2, w2 = left_img.shape[:2]

        # Get warped corners of right image
        corners_right = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners_right, H)
        all_corners = np.vstack([warped_corners.reshape(-1, 2),
                                 [[0, 0], [w2, 0], [w2, h2], [0, h2]]])

        [xmin, ymin] = np.int32(all_corners.min(axis=0) - 10)
        [xmax, ymax] = np.int32(all_corners.max(axis=0) + 10)

        # Translation matrix to shift everything positive
        t = [-xmin, -ymin]
        T = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        H_adj = T @ H

        # Warp right image
        result_warp = cv2.warpPerspective(right_img, H_adj, (xmax - xmin, ymax - ymin))

        # Create canvas for left image placement
        result = np.zeros((ymax - ymin, xmax - xmin, 3), dtype=np.uint8)
        result[t[1]:t[1]+h2, t[0]:t[0]+w2] = left_img

        # Blend overlapping region
        overlap_mask = (result > 0) & (result_warp > 0)
        if overlap_mask.any():
            # Simple linear blend in overlap zone
            blended = cv2.addWeighted(result_warp, 0.5, result, 0.5, 0)
            result = np.where(overlap_mask, blended, np.maximum(result, result_warp))
        else:
            result = np.maximum(result, result_warp)

        return result

    def _draw_sidebar(self, frame):
        """Overlay help panel on the left side."""
        if not self.show_help:
            return frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (360, frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)  # More opaque than before

        y = 30
        for line in self.help_lines:
            if not line.strip():
                y += 10
            else:
                draw_text(frame, line, (10, y), scale=0.45, color=(0, 255, 0))
                y += 22

        # Status indicators
        y += 20
        draw_text(frame, f"Mode: {self.mode}", (10, y), 0.6); y += 25
        draw_text(frame, f"Contrast: {self.contrast:.2f}", (10, y), 0.5); y += 20
        draw_text(frame, f"Brightness: {self.brightness}", (10, y), 0.5); y += 22

        angle = self.rotation_angles[self.rotation_angle_index]
        if angle != 0:
            draw_text(frame, f"Rotation: {angle}°", (10, y), 0.5); y += 20
        if abs(self.current_scale - 1.0) > 0.01:
            draw_text(frame, f"Scale: {self.current_scale:.2f}x", (10, y), 0.5); y += 20

        if self.mode == "gaussian":
            draw_text(frame, f"Kernel: {self.gaussian_kernel}", (10, y), 0.5)
        elif self.mode == "bilateral":
            p = self.bilateral_params
            draw_text(frame, f"d:{p['d']} sc:{p['sigmaColor']} ss:{p['sigmaSpace']}", (10, y), 0.45)
        return frame

    def run(self):
        """Main loop: capture -> process -> display -> handle input."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            output = self._process_frame(frame)
            output = self._draw_sidebar(output)

            if len(self.panorama_frames) > 0 and self.mode not in ["panorama", "ar", "calibration_result", "histogram"]:
                draw_text(output, f"Frames: {len(self.panorama_frames)}. Press 'p' to stitch.", (50, 50), 0.8, (255, 255, 0))

            cv2.imshow(self.window_name, output)

            key = cv2.waitKey(1) & 0xFF
            if self._handle_keypress(key, frame):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        """Apply current effect and geometric transforms."""
        adjusted = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)

        base_handlers = {
            'raw': lambda f: f,
            'gray': lambda f: cv2.cvtColor(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
            'hsv': lambda f: cv2.cvtColor(f, cv2.COLOR_BGR2HSV),
            'gaussian': self._apply_gaussian_blur,
            'bilateral': self._apply_bilateral_filter,
            'canny': lambda f: cv2.cvtColor(cv2.Canny(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), 100, 200), cv2.COLOR_GRAY2BGR),
            'hough': self._draw_hough_lines,
            'calibrate': self._run_calibration_step,
            'calibration_result': self._show_calibration_summary,
            'undistort': self._apply_undistort,
            'panorama': self._display_panorama,
            'ar': self._augment_reality,
            'translate': self._apply_fixed_translation,
            'histogram': self._show_histogram,  # New mode
        }

        proc = base_handlers.get(self.mode, lambda f: f)(adjusted)

        # Apply persistent transforms only if not in special view modes
        if self.mode not in ['panorama', 'calibration_result', 'histogram']:
            if self.current_scale != 1.0:
                proc = self._apply_scaling(proc)
            current_rotation = self.rotation_angles[self.rotation_angle_index]
            if current_rotation != 0:
                proc = self._apply_rotation(proc, current_rotation)

        return proc

    def _apply_gaussian_blur(self, frame):
        k = self.gaussian_kernel
        k = max(1, k if k % 2 == 1 else k + 1)
        return cv2.GaussianBlur(frame, (k, k), 0)

    def _apply_bilateral_filter(self, frame):
        p = self.bilateral_params
        return cv2.bilateralFilter(frame, d=p['d'], sigmaColor=p['sigmaColor'], sigmaSpace=p['sigmaSpace'])

    def _draw_hough_lines(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
        if lines is not None:
            for [[x1, y1, x2, y2]] in lines:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return frame

    def _apply_fixed_translation(self, frame):
        M = np.float32([[1, 0, 50], [0, 1, 30]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

    def _apply_rotation(self, frame, angle):
        h, w = frame.shape[:2]
        if angle == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def _apply_scaling(self, frame):
        h, w = frame.shape[:2]
        scaled = cv2.resize(frame, None, fx=self.current_scale, fy=self.current_scale, interpolation=cv2.INTER_LINEAR)
        sh, sw = scaled.shape[:2]
        if self.current_scale > 1.0:
            y0, x0 = (sh - h) // 2, (sw - w) // 2
            return scaled[y0:y0 + h, x0:x0 + w]
        else:
            canvas = np.full_like(frame, 128)
            y0, x0 = (h - sh) // 2, (w - sw) // 2
            canvas[y0:y0 + sh, x0:x0 + sw] = scaled
            return canvas

    def _show_histogram(self, frame):
        """Displays a large histogram overlaid on the frame with semi-transparent background."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, 250, cv2.NORM_MINMAX)

        # Create overlay
        overlay = np.zeros_like(frame)
        for i in range(1, 256):
            cv2.line(overlay, (i - 1, 250 - int(hist[i - 1][0])), (i, 250 - int(hist[i][0])), (0, 255, 0), 1)

        # Add label
        cv2.rectangle(overlay, (0, 0), (300, 40), (0, 0, 0), -1)
        draw_text(overlay, "Histogram", (10, 30), scale=0.7, color=(0, 255, 0))

        # Position histogram at bottom
        x_offset = (frame.shape[1] - 256) // 2
        y_offset = frame.shape[0] - 270

        # Blit histogram area
        roi = frame[y_offset:y_offset+270, x_offset:x_offset+256]
        blended = cv2.addWeighted(roi, 0.4, overlay[:270, :256], 0.6, 0)
        frame[y_offset:y_offset+270, x_offset:x_offset+256] = blended

        return frame

    def _run_calibration_step(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.CHESSBOARD_SIZE, flags=cv2.CALIB_CB_FAST_CHECK)
        if ret:
            cv2.drawChessboardCorners(frame, self.CHESSBOARD_SIZE, corners, ret)
            if time.time() - self.last_capture_time > 2 and len(self.objpoints) < self.TARGET_IMAGES:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
                self.objpoints.append(self.objp.copy())
                self.imgpoints.append(refined)
                print(f"Captured calibration image {len(self.objpoints)}/{self.TARGET_IMAGES}")
                self.last_capture_time = time.time()
        draw_text(frame, f"Calibrating... {len(self.objpoints)}/{self.TARGET_IMAGES}", (50, 50), 0.8, (0, 200, 255))
        if len(self.objpoints) >= self.TARGET_IMAGES:
            print("Running final calibration...")
            _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            mean_error = 0
            for i in range(len(self.objpoints)):
                proj_pts, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(self.imgpoints[i], proj_pts, cv2.NORM_L2) / len(proj_pts)
                mean_error += error
            final_error = mean_error / len(self.objpoints)
            self.mtx, self.dist = mtx, dist
            self.calibration_results = {'mtx': mtx, 'dist': dist, 'error': final_error}
            np.savez('calibration.npz', mtx=mtx, dist=dist, error=final_error)
            print(f"Calibration complete. Reprojection error: {final_error:.4f}")
            self.mode = "calibration_result"
        return frame

    def _show_calibration_summary(self, frame):
        disp = np.zeros_like(frame)
        res = self.calibration_results
        y = 60
        draw_text(disp, "CALIBRATION COMPLETE", (50, y), 1.0, (0, 255, 0)); y += 60
        draw_text(disp, f"Reprojection Error: {res.get('error', 0):.4f}", (60, y), 0.7); y += 50
        draw_text(disp, "Intrinsic Matrix:", (60, y), 0.6); y += 30
        if 'mtx' in res:
            for row in res['mtx']:
                draw_text(disp, f"[{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f}]", (80, y), 0.55); y += 30
        y += 10
        draw_text(disp, "Press 'u' for undistorted view.", (60, y), 0.7, (255, 255, 0))
        return disp

    def _apply_undistort(self, frame):
        if self.mtx is None:
            draw_text(frame, "Calibrate first with [k]", (50, 50), 0.8, (0, 0, 255))
            return frame
        h, w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, self.mtx, self.dist, None, new_mtx)
        x, y, ww, hh = roi
        if ww > 0 and hh > 0:
            undistorted = undistorted[y:y + hh, x:x + ww]
        return cv2.resize(undistorted, (w, h))

    def _display_panorama(self, frame):
        if self.panorama_result is not None:
            h_target, w_target = frame.shape[:2]
            resized = cv2.resize(self.panorama_result, (w_target, h_target))
            draw_text(resized, "Panorama Mode - 'z' to reset", (50, 50), 0.8, (255, 255, 0))
            return resized
        draw_text(frame, "Capture frames [o], then stitch [p]", (50, 50), 0.7, (255, 255, 0))
        return frame

    def _augment_reality(self, frame):
        if len(self.model_vertices) == 0:
            draw_text(frame, "3D model not loaded.", (20, 90), 0.8, (0, 0, 255))
            return frame
        if self.mtx is None or self.dist is None:
            draw_text(frame, "Calibration required.", (20, 90), 0.8, (0, 165, 255))
            return frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if hasattr(cv2.aruco, 'ArucoDetector'):
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, self.mtx, self.dist)
            for i in range(len(ids)):
                pts, _ = cv2.projectPoints(self.model_vertices, rvecs[i][0], tvecs[i][0], self.mtx, self.dist)
                for face in self.model_faces:
                    poly = np.int32(pts[face]).reshape(-1, 2)
                    cv2.polylines(frame, [poly], True, (0, 255, 128), 1, cv2.LINE_AA)
        else:
            draw_text(frame, "No ArUco markers detected", (20, 90), 0.8, (0, 0, 255))
        return frame

    def _handle_keypress(self, key, frame):
        """Process keyboard input."""
        if key == ord('q'):
            return True  # Exit

        # Mode switches
        mode_map = {
            '0': 'raw', '1': 'gray', '2': 'hsv',
            'g': 'gaussian', 'f': 'bilateral',
            'e': 'canny', 'l': 'hough',
            't': 'translate', 'a': 'ar',
            'k': 'calibrate', 'u': 'undistort',
            'h': 'histogram'
        }
        ch = chr(key) if 0 <= key < 256 else ''
        if ch in mode_map:
            if ch == 'k':
                self.objpoints.clear(); self.imgpoints.clear()
            self.mode = mode_map[ch]

        # Toggle help with '?'
        elif key == ord('?'):
            self.show_help = not self.show_help

        # Rotation
        elif key == ord('r'):
            self.rotation_angle_index = (self.rotation_angle_index + 1) % 4

        # Brightness/Contrast
        elif key == ord('c'): self.contrast = min(3.0, self.contrast + 0.1)
        elif key == ord('v'): self.contrast = max(0.1, self.contrast - 0.1)
        elif key == ord('b'): self.brightness = min(100, self.brightness + 5)
        elif key == ord('n'): self.brightness = max(-100, self.brightness - 5)

        # Scale control
        elif key == ord('s'):
            self.scale_index = (self.scale_index + 1) % len(self.scales)
            self.current_scale = self.scales[self.scale_index]
        elif key == ord(']'):
            self.current_scale = min(4.0, self.current_scale * 1.1)
        elif key == ord('['):
            self.current_scale = max(0.1, self.current_scale * 0.9)

        # Gaussian
        elif self.mode == 'gaussian':
            if key in (ord('+'), ord('=')): self.gaussian_kernel += 2
            elif key == ord('-'): self.gaussian_kernel = max(1, self.gaussian_kernel - 2)

        # Bilateral
        elif self.mode == 'bilateral':
            if key == ord('d'): self.bilateral_params['d'] = max(1, self.bilateral_params['d'] - 2)
            elif key == ord('D'): self.bilateral_params['d'] += 2
            elif key == ord('m'): self.bilateral_params['sigmaColor'] = max(1, self.bilateral_params['sigmaColor'] - 5)
            elif key == ord('M'): self.bilateral_params['sigmaColor'] += 5
            elif key == ord('x'): self.bilateral_params['sigmaSpace'] = max(1, self.bilateral_params['sigmaSpace'] - 5)
            elif key == ord('X'): self.bilateral_params['sigmaSpace'] += 5

        # Panorama
        elif key == ord('o'):
            self.panorama_frames.append(frame.copy())
            print(f"Captured image {len(self.panorama_frames)} for panorama")
        elif key == ord('p'):
            if len(self.panorama_frames) < 2:
                print("Need at least 2 images to stitch.")
            else:
                print("Stitching panorama...")
                base = self.panorama_frames[0]
                for next_img in self.panorama_frames[1:]:
                    base = self._stitch_two_images(next_img, base)
                self.panorama_result = base
                self.mode = "panorama"
                print("Panorama stitching complete.")

        # Reset
        elif key == ord('z'):
            self.mode = 'raw'; self.contrast = 1.0; self.brightness = 0
            self.rotation_angle_index = 0
            self.scale_index = 1; self.current_scale = self.scales[self.scale_index]
            self.gaussian_kernel = 9
            self.bilateral_params = {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75}
            self.panorama_frames.clear(); self.panorama_result = None
            print("All settings reset to default.")

        return False


if __name__ == '__main__':
    app = LiveCameraEffects()
    app.run()