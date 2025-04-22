import argparse
import cv2
import os
import numpy as np
from datetime import datetime

def calculate_sharpness(image):
    """Calculate sharpness metric using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_frames(video_path, output_dir, interval_sec=0.15, max_frames=None, 
                  target_size=None, min_sharpness=20, min_brightness=30):
    """
    Extract frames from a video with quality filtering and resizing.
    
    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        interval_sec (float): Interval in seconds between extracted frames.
        max_frames (int): Maximum number of frames to extract.
        target_size (tuple): Target (width, height) for resizing (None to keep original).
        min_sharpness (float): Minimum sharpness threshold (discard blurry frames).
        min_brightness (float): Minimum brightness threshold (discard dark frames).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    frame_interval = int(fps * interval_sec)
    
    print(f"Video Info: {fps} FPS, {total_frames} frames, {duration_sec:.2f} seconds")
    print(f"Extracting 1 frame every {interval_sec} seconds ({frame_interval} frames)")
    if target_size:
        print(f"Resizing frames to {target_size[0]}x{target_size[1]}")
    print(f"Quality thresholds - Sharpness: {min_sharpness}, Brightness: {min_brightness}\n")
    
    frame_count = 0
    saved_count = 0
    rejected_count = 0
    timestamps = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame at the specified interval
        if frame_count % frame_interval == 0:
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Calculate quality metrics
            sharpness = calculate_sharpness(frame)
            brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[...,2])
            
            # Check quality thresholds
            if sharpness >= min_sharpness and brightness >= min_brightness:
                # Resize if target size specified
                if target_size is not None:
                    # Calculate scaling factor while maintaining aspect ratio
                    h, w = frame.shape[:2]
                    target_w, target_h = target_size
                    
                    # Determine which dimension to scale by
                    scale = min(target_w/w, target_h/h)
                    
                    # Resize while maintaining aspect ratio
                    resized = cv2.resize(frame, (int(w*scale), int(h*scale)), 
                                        interpolation=cv2.INTER_AREA)
                    
                    # Calculate padding or cropping
                    dw = target_w - resized.shape[1]
                    dh = target_h - resized.shape[0]
                    
                    # Center crop (remove this if you prefer padding)
                    if dw < 0:  # Width is larger than target
                        x = int(-dw/2)
                        resized = resized[:, x:x+target_w]
                    if dh < 0:  # Height is larger than target
                        y = int(-dh/2)
                        resized = resized[y:y+target_h, :]
                    
                    frame = resized
                
                # Generate filename with quality metrics
                filename = os.path.join(
                    output_dir, 
                    f"frame_{saved_count:03d}_{timestamp:.2f}s_" +
                    f"sh{sharpness:.1f}_br{brightness:.1f}.jpg"
                )
                
                cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
                timestamps.append(timestamp)
                print(f"✅ Saved {filename}")
            else:
                rejected_count += 1
                print(f"❌ Rejected frame {frame_count} (sh: {sharpness:.1f}, br: {brightness:.1f})")
            
            # Stop if we've reached max_frames
            if max_frames is not None and saved_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    
    # Print summary
    print(f"\nExtraction complete:")
    print(f"- Saved {saved_count} frames to {output_dir}")
    print(f"- Rejected {rejected_count} frames for quality")
    if saved_count > 0:
        print(f"- Average sharpness: {np.mean([float(f.split('_sh')[1].split('_')[0]) for f in os.listdir(output_dir) if f.endswith('.jpg')]):.1f}")
        print(f"- Average brightness: {np.mean([float(f.split('_br')[1].split('.jpg')[0]) for f in os.listdir(output_dir) if f.endswith('.jpg')]):.1f}")
    
    # Save timestamps to a file
    timestamp_file = os.path.join(output_dir, "extraction_report.txt")
    with open(timestamp_file, 'w') as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Frames extracted: {saved_count}\n")
        f.write(f"Frames rejected: {rejected_count}\n")
        f.write(f"Interval: {interval_sec} seconds\n")
        f.write(f"Target size: {target_size}\n")
        f.write(f"Min sharpness: {min_sharpness}\n")
        f.write(f"Min brightness: {min_brightness}\n\n")
        f.write("Frame timestamps:\n")
        for ts in timestamps:
            f.write(f"{ts:.2f}\n")
    print(f"\nExtraction report saved to {timestamp_file}")
    
    pass

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract frames from video for YOLO training')
    parser.add_argument('-i', '--input', required=True, help='Input video file path')
    parser.add_argument('-o', '--output', required=True, help='Output directory for frames')
    parser.add_argument('--interval', type=float, default=1.0, 
                       help='Interval between frames in seconds (default: 1.0)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to extract (default: unlimited)')
    parser.add_argument('--size', type=int, nargs=2, default=None,
                       metavar=('WIDTH', 'HEIGHT'), 
                       help='Target size for resizing (e.g., 640 640)')
    parser.add_argument('--min-sharpness', type=float, default=20.0,
                       help='Minimum sharpness threshold (default: 20.0)')
    parser.add_argument('--min-brightness', type=float, default=30.0,
                       help='Minimum brightness threshold (default: 30.0)')
    
    args = parser.parse_args()
    
    # Convert size tuple if provided
    target_size = tuple(args.size) if args.size else None
    
    # Run extraction
    extract_frames(
        video_path=args.input,
        output_dir=args.output,
        interval_sec=args.interval,
        max_frames=args.max_frames,
        target_size=target_size,
        min_sharpness=args.min_sharpness,
        min_brightness=args.min_brightness
    )