import h5py
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from segment_anything import sam_model_registry, SamPredictor
from ultralytics.trackers.byte_tracker import BYTETracker
from dataclasses import dataclass
import argparse
import os
import click
import glob
import re
from pathlib import Path

from tqdm import tqdm

class SAMObjectSegmentationTracker:
    def __init__(self, hdf5_path, sam_checkpoint, image_key='front_image', model_type="vit_h", device="cuda", 
                 save_visualization=True, start_index=0, collect_all_inputs_first=False,
                 high_res_dir=None, camera_name='front_rgb'):
        """
        Initialize the SAM-based object segmentation and tracking system.
        
        Args:
            hdf5_path: Path to the HDF5 dataset
            sam_checkpoint: Path to the SAM model checkpoint
            model_type: SAM model type (default: vit_h)
            device: Device to run the model on (default: cuda)
            save_visualization: Whether to save visualization (default: True)
            start_index: Index to start from (default: 0)
            collect_all_inputs_first: Whether to collect all inputs first, then process all demos (default: False)
            high_res_dir: Directory containing high-resolution images (default: None)
            camera_name: Camera name in the high-resolution directory (default: front_rgb)
        """
        self.hdf5_path = hdf5_path
        self.device = device
        
        # Initialize SAM
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        
        @dataclass
        class BYTETrackerArgs:
            track_low_thresh: float = 0.4
            track_high_thresh: float = 0.8
            new_track_thresh: float = 0.8
            fuse_score: bool = False
            track_buffer: int = 30
            match_thresh: float = 0.8
        
        # Initialize object tracker
        self.tracker = BYTETracker(
            args=BYTETrackerArgs(),
            frame_rate=30
        )
        
        self.image_key = image_key
        self.save_visualization = save_visualization
        self.start_index = start_index
        self.collect_all_inputs_first = collect_all_inputs_first
        
        # High-resolution directory and camera name
        self.high_res_dir = high_res_dir
        self.camera_name = camera_name
        
        # Storage for clicked points and masks
        self.points = []
        self.point_markers = []  # Store references to point markers for removal
        self.point_labels = []   # Store references to point labels for removal
        self.masks = {}          # Resized masks for storage (or original if not high-res)
        self.tracking_masks = {} # High-res masks for tracking (same as masks if not high-res)
        self.current_demo = None
        self.current_frame = None
        self.output_dir = None
        self.dataset = None
        # New storage for points per demo
        self.demo_points = {}
        
        # Cache for high-res demo paths
        self.high_res_demo_paths = None

    def __enter__(self):
        """Open the HDF5 file when entering the context."""
        self.dataset = h5py.File(self.hdf5_path, 'r+')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the HDF5 file when exiting the context."""
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
    
    def get_high_res_demo_paths(self):
        """
        Get list of high-resolution demo paths in alphanumerical order.
        
        Returns:
            List of paths to high-resolution demo directories, in alphanumerical order
        """
        if self.high_res_demo_paths is not None:
            return self.high_res_demo_paths
            
        if not self.high_res_dir:
            return []
            
        # Get all demo directories in alphanumerical order (demo_10 comes after demo_1 but before demo_2)
        demo_dirs = sorted(glob.glob(os.path.join(self.high_res_dir, "demo_*")), 
                           key=lambda x: [int(c) if c.isdigit() else c for c in re.findall(r'[^0-9]|[0-9]+', os.path.basename(x))])
        self.high_res_demo_paths = demo_dirs
        return self.high_res_demo_paths
    
    def get_high_res_image_path(self, demo_idx, frame_idx):
        """
        Get the path to the high-resolution image for the given demo index and frame.
        
        Args:
            demo_idx: The demo index (0-based) from the HDF5 file
            frame_idx: The frame index
            
        Returns:
            Path to the high-resolution image
        """
        if not self.high_res_dir:
            return None
            
        # Get demo paths in alphanumerical order
        demo_paths = self.get_high_res_demo_paths()
                
        # Check if demo index is valid
        if demo_idx >= len(demo_paths):
            return None
                    
        # Get the demo directory
        demo_dir = demo_paths[demo_idx]
        
        # Format the image file name
        image_file = f"image_{frame_idx+1:04d}.jpg"
        
        # Construct the full path
        image_path = os.path.join(demo_dir, self.camera_name, image_file)
        
        if os.path.exists(image_path):
            return image_path
        
        # If the path doesn't exist, return None
        return None
    
    def load_image(self, demo_key, frame_idx):
        """
        Load the image for the given demo and frame, preferring high-resolution if available.
        
        Args:
            demo_key: The demo key
            frame_idx: The frame index
            
        Returns:
            The loaded image and a flag indicating if it's high resolution
        """
        # Get the demo index in the HDF5 file
        demo_keys = list(self.dataset['data'].keys())
        
        # Sort by numerical part of the demo key
        def get_demo_number(key):
            match = re.search(r'demo_(\d+)', key)
            return int(match.group(1)) if match else 0
        
        demo_idx = get_demo_number(demo_key)
        
        # Try to load high-resolution image first
        high_res_path = self.get_high_res_image_path(demo_idx, frame_idx)
        
        if high_res_path and os.path.exists(high_res_path):
            print(f"Loading high-resolution image from {high_res_path}")
            # Load high-resolution image
            image = cv2.imread(high_res_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, True
        
        # Fall back to HDF5 image
        image = self.dataset['data'][demo_key]['obs'][self.image_key][frame_idx]
        
        # Ensure image is in correct format
        if image.shape[-1] != 3:  # Check if channels are not last dimension
            image = np.transpose(image, (1, 2, 0))
            
        return image, False
    
    def process_dataset(self, output_dir="output_masks"):
        """
        Process all demos in the dataset.
        
        Args:
            output_dir: Directory to save output masks
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all demo keys and sort them
        demo_keys = sorted(self.dataset['data'].keys())
        
        # Filter demo keys based on start_index
        filtered_demo_keys = []
        for i, demo_key in enumerate(demo_keys):
            if i < self.start_index:
                continue
            filtered_demo_keys.append(demo_key)
        
        if self.collect_all_inputs_first:
            # Two-phase approach: Collect all inputs first, then process all demos
            print(f"Phase 1: Collecting user inputs for {len(filtered_demo_keys)} demos...")
            self.collect_all_user_inputs(filtered_demo_keys)
            
            print(f"Phase 2: Processing all demos...")
            for i, demo_key in enumerate(filtered_demo_keys):
                print(f"Index: {i+self.start_index}, Processing {demo_key}...")
                self.process_demo_with_points(demo_key)
        else:
            # Original approach: Process each demo completely before moving to the next
            for i, demo_key in enumerate(filtered_demo_keys):
                print(f"Index: {i+self.start_index}, Processing {demo_key}...")
                self.process_demo(demo_key)
    
    def collect_all_user_inputs(self, demo_keys):
        """
        Collect user inputs for all demos first.
        
        Args:
            demo_keys: List of demo keys to collect inputs for
        """
        for i, demo_key in enumerate(demo_keys):
            print(f"Please select points for index {i + self.start_index}, {demo_key}...")
            self.collect_points_for_demo(demo_key)
    
    def collect_points_for_demo(self, demo_key):
        """
        Collect points for a single demo.
        
        Args:
            demo_key: Key of the demo to collect points for
        """
        self.current_demo = demo_key
        
        # Get the first frame (try high-res first)
        first_frame, is_high_res = self.load_image(demo_key, 0)
        self.current_frame = first_frame
        
        # Reset points for this demo
        self.points = []
        self.point_markers = []
        self.point_labels = []
        
        # Display UI for user to click on points
        self.display_point_selection_ui(first_frame, is_high_res)
        
        # Store the points for this demo
        self.demo_points[demo_key] = {
            'points': self.points.copy(),
            'is_high_res': is_high_res
        }
    
    def process_demo_with_points(self, demo_key):
        """
        Process a single demo using previously collected points.
        
        Args:
            demo_key: Key of the demo to process
        """
        self.current_demo = demo_key
        demo_data = self.dataset['data'][demo_key]
        self.tracker.reset()
        
        # Create output directory for this demo
        demo_output_dir = os.path.join(self.output_dir, demo_key)
        os.makedirs(demo_output_dir, exist_ok=True)
        
        # Get the first frame
        first_frame, is_high_res = self.load_image(demo_key, 0)
        self.current_frame = first_frame
        
        # Get HDF5 frame dimensions for later rescaling
        hdf5_frame_shape = demo_data['obs'][self.image_key][0].shape
        if len(hdf5_frame_shape) == 3 and hdf5_frame_shape[0] == 3:  # If channels first
            hdf5_height, hdf5_width = hdf5_frame_shape[1], hdf5_frame_shape[2]
        else:  # If channels last
            hdf5_height, hdf5_width = hdf5_frame_shape[0], hdf5_frame_shape[1]
        
        # Reset masks
        self.masks = {}
        self.tracking_masks = {}
        
        # Retrieve stored points for this demo
        stored_points = self.demo_points[demo_key]
        self.points = stored_points['points']
        
        # Generate masks for the first frame
        self.generate_masks_for_first_frame(hdf5_height, hdf5_width, is_high_res)
    
        self.all_masks = []
        # Save the first frame masks
        if self.save_visualization:
            # Always use HDF5 image for visualization
            hdf5_image = self.dataset['data'][demo_key]['obs'][self.image_key][0]
            if hdf5_image.shape[-1] != 3:  # Check if channels are not last dimension
                hdf5_image = np.transpose(hdf5_image, (1, 2, 0))
            self.save_mask_vis(0, demo_output_dir, hdf5_image)
        self.all_masks.append(self.masks)
        
        # Process remaining frames
        num_frames = len(demo_data['obs'][self.image_key])
        for frame_idx in tqdm(range(1, num_frames), desc=f"Tracking objects in {demo_key}"):
            frame, frame_is_high_res = self.load_image(demo_key, frame_idx)
            self.track_objects_in_frame(frame, frame_idx, hdf5_height, hdf5_width, frame_is_high_res)
            self.all_masks.append(self.masks)
            if self.save_visualization:
                # Always use HDF5 image for visualization
                hdf5_image = self.dataset['data'][demo_key]['obs'][self.image_key][frame_idx]
                if hdf5_image.shape[-1] != 3:  # Check if channels are not last dimension
                    hdf5_image = np.transpose(hdf5_image, (1, 2, 0))
                self.save_mask_vis(frame_idx, demo_output_dir, hdf5_image)
                
        self.save_masks_to_dataset(demo_key, self.all_masks)
    
    def process_demo(self, demo_key):
        """
        Process a single demo with user input.
        
        Args:
            demo_key: Key of the demo to process
        """
        self.current_demo = demo_key
        demo_data = self.dataset['data'][demo_key]
        self.tracker.reset()
        
        # Create output directory for this demo
        demo_output_dir = os.path.join(self.output_dir, demo_key)
        os.makedirs(demo_output_dir, exist_ok=True)
        
        # Get the first frame (try high-res first)
        first_frame, is_high_res = self.load_image(demo_key, 0)
        self.current_frame = first_frame
        
        # Get HDF5 frame dimensions for later rescaling
        hdf5_frame_shape = demo_data['obs'][self.image_key][0].shape
        if len(hdf5_frame_shape) == 3 and hdf5_frame_shape[0] == 3:  # If channels first
            hdf5_height, hdf5_width = hdf5_frame_shape[1], hdf5_frame_shape[2]
        else:  # If channels last
            hdf5_height, hdf5_width = hdf5_frame_shape[0], hdf5_frame_shape[1]
        
        # Reset points and masks
        self.points = []
        self.point_markers = []
        self.point_labels = []
        self.masks = {}
        self.tracking_masks = {}
        
        # Display UI for user to click on points
        self.display_point_selection_ui(first_frame, is_high_res)
        
        # Generate masks for the first frame
        self.generate_masks_for_first_frame(hdf5_height, hdf5_width, is_high_res)
    
        self.all_masks = []
        # Save the first frame masks
        if self.save_visualization:
            # Always use HDF5 image for visualization
            hdf5_image = self.dataset['data'][demo_key]['obs'][self.image_key][0]
            if hdf5_image.shape[-1] != 3:  # Check if channels are not last dimension
                hdf5_image = np.transpose(hdf5_image, (1, 2, 0))
            self.save_mask_vis(0, demo_output_dir, hdf5_image)
        self.all_masks.append(self.masks)
        
        # Process remaining frames
        num_frames = len(demo_data['obs'][self.image_key])
        for frame_idx in tqdm(range(1, num_frames), desc=f"Tracking objects in {demo_key}"):
            frame, frame_is_high_res = self.load_image(demo_key, frame_idx)
            self.track_objects_in_frame(frame, frame_idx, hdf5_height, hdf5_width, frame_is_high_res)
            self.all_masks.append(self.masks)
            if self.save_visualization:
                # Always use HDF5 image for visualization
                hdf5_image = self.dataset['data'][demo_key]['obs'][self.image_key][frame_idx]
                if hdf5_image.shape[-1] != 3:  # Check if channels are not last dimension
                    hdf5_image = np.transpose(hdf5_image, (1, 2, 0))
                self.save_mask_vis(frame_idx, demo_output_dir, hdf5_image)
                
        self.save_masks_to_dataset(demo_key, self.all_masks)
            
    def display_point_selection_ui(self, frame, is_high_res=False):
        """
        Display a UI for the user to click on points of interest in the first frame.
        
        Args:
            frame: The first frame of the demo
            is_high_res: Whether this is a high-resolution image
        """
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.imshow(frame)
        resolution_info = "High Resolution" if is_high_res else "Low Resolution (from HDF5)"
        self.ax.set_title(f"Click on objects of interest in {self.current_demo} [{resolution_info}]\nShift+Click to remove a point\nPress 'Done' when finished")
        
        # Add a button to finish selection
        done_ax = plt.axes([0.85, 0.01, 0.1, 0.075])
        self.done_button = Button(done_ax, 'Done')
        self.done_button.on_clicked(self.on_done)
        
        # Connect click event
        self.click_connection = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()
    
    def on_click(self, event):
        """
        Handle click event in the UI.
        
        Args:
            event: MouseEvent from matplotlib
        """
        if event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            
            # Check if shift key is pressed (for removing points)
            if event.key == 'shift':
                # Find the closest point to remove
                if self.points:
                    distances = [np.sqrt((p[0] - x)**2 + (p[1] - y)**2) for p in self.points]
                    closest_idx = np.argmin(distances)
                    
                    # Only remove if within a reasonable distance (e.g., 20 pixels)
                    if distances[closest_idx] < 20:
                        # Remove the point
                        self.points.pop(closest_idx)
                        
                        # Remove the marker and label
                        if closest_idx < len(self.point_markers):
                            marker = self.point_markers.pop(closest_idx)
                            marker.remove()
                        
                        if closest_idx < len(self.point_labels):
                            label = self.point_labels.pop(closest_idx)
                            label.remove()
                        
                        # Redraw remaining points with updated indices
                        for i, marker in enumerate(self.point_markers):
                            marker.set_color('red')
                        
                        for i, label in enumerate(self.point_labels):
                            label.set_text(f"{i+1}")
                        
                        self.fig.canvas.draw()
            else:
                # Add a new point
                self.points.append([x, y])
                
                # Mark the point on the image
                marker = self.ax.plot(x, y, 'ro', markersize=10)[0]
                label = self.ax.text(x+5, y+5, f"{len(self.points)}", color='white', 
                             bbox=dict(facecolor='red', alpha=0.5))
                
                # Store references to the marker and label
                self.point_markers.append(marker)
                self.point_labels.append(label)
                
                self.fig.canvas.draw()
    
    def on_done(self, event):
        """
        Handle done button press.
        
        Args:
            event: ButtonEvent from matplotlib
        """
        plt.close(self.fig)
        
    def save_masks_to_dataset(self, demo_key, all_masks):
        """
        Save masks to the dataset.
        
        Args:
            demo_key: Key of the demo
            all_masks: List of masks to save
        """
        # Check if mask group already exists, if so delete it
        if 'object_masks' in self.dataset['data'][demo_key]:
            del self.dataset['data'][demo_key]['object_masks']
        
        # Create the mask group
        mask_group = self.dataset['data'][demo_key].create_group('object_masks')

        for frame_idx, masks in enumerate(all_masks):
            # Check if frame group already exists, if so delete it
            frame_group_name = f'frame_{frame_idx}'
            if frame_group_name in mask_group:
                del mask_group[frame_group_name]
            
            # Create new group for this frame
            frame_group = mask_group.create_group(frame_group_name)
            
            for mask_id, mask in masks.items():
                # Check if mask already exists, if so delete it
                mask_name = f'mask_{mask_id}'
                if mask_name in frame_group:
                    del frame_group[mask_name]
                
                # Save mask
                frame_group.create_dataset(mask_name, data=mask)
    
    def resize_mask(self, mask, target_height, target_width):
        """
        Resize a mask to target dimensions.
        
        Args:
            mask: Binary mask to resize
            target_height: Target height
            target_width: Target width
            
        Returns:
            Resized binary mask
        """
        resized_mask = cv2.resize(
            mask.astype(np.uint8), 
            (target_width, target_height), 
            interpolation=cv2.INTER_AREA
        )
        return resized_mask.astype(bool)
    
    def generate_masks_for_first_frame(self, hdf5_height=None, hdf5_width=None, is_high_res=False):
        """
        Generate masks for all clicked points in the first frame using SAM.
        
        Args:
            hdf5_height: Height of frames in HDF5
            hdf5_width: Width of frames in HDF5
            is_high_res: Whether the current frame is high resolution
        """        
        if not self.points:
            print("No points selected. Please select at least one point.")
            return
                
        # Set the image in the predictor
        self.predictor.set_image(self.current_frame)
                        
        # Generate masks for each point
        for i, point in enumerate(self.points):
            input_point = np.array([point])
            input_label = np.array([1])  # 1 indicates foreground
            
            # Generate mask
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            # Select the mask with the highest score
            best_mask_idx = np.argmax(scores)
            mask = masks[best_mask_idx]
            
            # Store the original mask for tracking
            self.tracking_masks[i+1] = mask.copy()
            
            # If using high-res images, resize the mask to match HDF5 dimensions for storage
            if is_high_res and hdf5_height is not None and hdf5_width is not None:
                mask = self.resize_mask(mask, hdf5_height, hdf5_width)
            
            # Store the mask with an ID (resized or original for storage)
            self.masks[i+1] = mask  # ID starts from 1
            
            print(f"Generated mask for point {i+1} with confidence {scores[best_mask_idx]:.4f}")
    
    def track_objects_in_frame(self, frame, frame_idx, hdf5_height=None, hdf5_width=None, is_high_res=False):
        """
        Track objects from previous frame to current frame.
        
        Args:
            frame: Current frame
            frame_idx: Index of the current frame
            hdf5_height: Height of frames in HDF5
            hdf5_width: Width of frames in HDF5
            is_high_res: Whether the current frame is high resolution
        """
        if not self.masks:
            print("No masks to track. Please generate masks for the first frame first.")
            return
        
        # Set the image in the predictor
        self.predictor.set_image(frame)
        
        # Convert masks to detections for tracking
        # For BYTETracker, we need to format detections as results object with xywh, conf, and cls
        bboxes = []
        scores = []
        cls = []
        mask_ids = []
        
        # Use tracking_masks for computing bounding boxes if high-res, otherwise use masks
        tracking_masks = self.tracking_masks if is_high_res else self.masks
        
        for mask_id, mask in tracking_masks.items():
            # Find contours of the mask
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Store in xywh format (center_x, center_y, width, height)
                cx = x + w/2
                cy = y + h/2
                bboxes.append([cx, cy, w, h])
                
                # Calculate confidence (placeholder, can be improved)
                scores.append(0.9)
                
                # Store class (all objects are the same class in this case)
                cls.append(0)
                
                # Store mask_id for reference
                mask_ids.append(mask_id)
        
        if bboxes:
            # Create a Results-like object for BYTETracker
            class Results:
                pass
            
            results = Results()
            results.xywh = np.array(bboxes)
            results.conf = np.array(scores)
            results.cls = np.array(cls)
                        
            # Track objects
            tracks = self.tracker.update(results, frame)
                        
            # Update masks based on tracking results
            new_masks = {}       # For storage (resized if high-res)
            new_tracking_masks = {}  # For tracking (original high-res)
            
            # Create a mapping from track_id to original mask_id
            # This is necessary because BYTETracker may assign different track_ids
            track_to_mask_map = {}
            
            # We need to match the tracks back to our original mask_ids
            # Use IOU matching to find the best correspondences
            for i, track in enumerate(tracks):
                if len(track) >= 4:  # Make sure we have at least x, y, w, h
                    # Extract the bbox from the track result
                    t_left, t_top, t_right, t_bottom = track[:4]
                    
                    # This track's box as [left, top, right, bottom]
                    track_box = [t_left, t_top, t_right, t_bottom]
                    
                    # Find the best matching mask_id based on IOU
                    best_iou = 0
                    best_mask_id = None
                    
                    for j, mask_id in enumerate(mask_ids):
                        # Original box
                        x, y, w, h = bboxes[j]
                        left, top = x - w/2, y - h/2
                        right, bottom = x + w/2, y + h/2
                        
                        # Calculate IOU
                        x_left = max(left, track_box[0])
                        y_top = max(top, track_box[1])
                        x_right = min(right, track_box[2])
                        y_bottom = min(bottom, track_box[3])
                        
                        if x_right < x_left or y_bottom < y_top:
                            continue
                        
                        intersection = (x_right - x_left) * (y_bottom - y_top)
                        box1_area = (right - left) * (bottom - top)
                        box2_area = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
                        union = box1_area + box2_area - intersection
                        
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_mask_id = mask_id
                    
                    if best_mask_id is not None and best_iou > 0.3:  # IOU threshold
                        track_to_mask_map[i] = best_mask_id
            
            # Now generate new masks based on the tracks
            for i, track in enumerate(tracks):
                if i in track_to_mask_map:
                    mask_id = track_to_mask_map[i]
                    
                    # Get the previous mask's area (from tracking masks)
                    prev_mask = tracking_masks[mask_id]
                    prev_area = np.sum(prev_mask)
                    
                    # Extract bbox from track result
                    box = track[:4]  # [left, top, right, bottom]
                    
                    # Calculate a point in the center of the box
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    
                    # Generate a new mask
                    input_point = np.array([[center_x, center_y]])
                    input_label = np.array([1])
                    
                    masks, scores, logits = self.predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                        box=np.array([box[0], box[1], box[2], box[3]])  # Provide box constraint
                    )
                    
                    # Find the mask that's closest in area to the previous mask
                    best_mask_idx = 0
                    best_area_diff = float('inf')
                    
                    for idx, mask in enumerate(masks):
                        current_area = np.sum(mask)
                        area_diff = abs(current_area - prev_area) / prev_area  # Relative difference
                        
                        if area_diff < best_area_diff:
                            best_area_diff = area_diff
                            best_mask_idx = idx
                    
                    # Only use the new mask if it's not too different in size
                    new_mask = masks[best_mask_idx]
                    new_area = np.sum(new_mask)
                    area_ratio = new_area / prev_area
                    
                    # Always keep the high-res mask for tracking
                    new_tracking_mask = new_mask.copy()
                    
                    # If using high-res images, resize the mask to match HDF5 dimensions for storage
                    if is_high_res and hdf5_height is not None and hdf5_width is not None:
                        new_mask = self.resize_mask(new_mask, hdf5_height, hdf5_width)
                        # Recalculate area ratio for the storage decision, using original areas
                        storage_area = np.sum(new_mask)
                        storage_area_ratio = storage_area / np.sum(self.masks[mask_id])
                    else:
                        storage_area_ratio = area_ratio
                    
                    # Allow the mask to be between 0.5x and 2x the previous size
                    if 0.5 <= area_ratio <= 2.0:
                        new_tracking_masks[mask_id] = new_tracking_mask
                        new_masks[mask_id] = new_mask
                    else:
                        # If the new mask is too different, keep the previous masks
                        new_tracking_masks[mask_id] = tracking_masks[mask_id]
                        new_masks[mask_id] = self.masks[mask_id]
            
            # Update masks
            self.tracking_masks = new_tracking_masks
            self.masks = new_masks
    
    def save_mask_vis(self, frame_idx, output_dir, orig_image, mask_alpha=0.5):
        """
        Save masks for the current frame.
        
        Args:
            frame_idx: Index of the current frame
            output_dir: Directory to save masks
            orig_image: Original image (from HDF5)
            mask_alpha: Alpha value for mask transparency
        """
        # Save a visualization of all masks
        if self.masks:
            first_mask = next(iter(self.masks.values()))
            
            colored_mask = np.zeros((*first_mask.shape, 4), dtype=np.float32)

            # Generate unique colors for each mask ID
            num_masks = len(self.masks)
            # Use a colormap to generate distinct colors
            cmap = plt.get_cmap('hsv', num_masks + 1)
            
            for i, (mask_id, mask) in enumerate(self.masks.items()):
                # Get a unique color from the colormap
                color = cmap(i)
                # Set alpha value
                mask_color = (color[0], color[1], color[2], mask_alpha)
                colored_mask[mask] = mask_color
            
            # Save combined mask
            combined_path = os.path.join(output_dir, f"combined_mask_{frame_idx:04d}.png")
            plt.figure(figsize=(10, 10))
            plt.imshow(orig_image)
            plt.imshow(colored_mask)
            plt.title(f"Segmentation Masks - Frame {frame_idx}")
            plt.savefig(combined_path)
            plt.close()


def main():    
    @click.command()
    @click.option("--dataset", required=True, help="Path to HDF5 dataset")
    @click.option("--sam-checkpoint", required=True, help="Path to SAM checkpoint")
    @click.option("--model-type", default="vit_b", help="SAM model type")
    @click.option("--device", default="cuda", help="Device to run on")
    @click.option("--save-visualization", default=True, help="Save visualization")
    @click.option("--output-dir", default="output_masks", help="Output directory for vis")
    @click.option("--image-key", default="front_image", help="Image key")
    @click.option("--start-index", default=0, help="Start index")
    @click.option("--collect-all-inputs-first", '-c',is_flag=True, help="Collect all user inputs first, then process all demos")
    @click.option("--high-res-dir", help="Directory containing high-resolution images")
    @click.option("--camera-name", default="front_rgb", help="Camera name in high-resolution directory")
    def run(dataset, sam_checkpoint, model_type, device, output_dir, save_visualization, image_key, start_index, collect_all_inputs_first, high_res_dir, camera_name):
        with SAMObjectSegmentationTracker(
            hdf5_path=dataset, 
            sam_checkpoint=sam_checkpoint,
            image_key=image_key,    
            model_type=model_type,
            device=device,
            save_visualization=save_visualization,
            start_index=start_index,
            collect_all_inputs_first=collect_all_inputs_first,
            high_res_dir=high_res_dir,
            camera_name=camera_name
        ) as tracker:
            tracker.process_dataset(output_dir)
    
    run()

if __name__ == "__main__":
    main()