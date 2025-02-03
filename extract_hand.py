import cv2
import os

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return
    
    # Get the video frame rate (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Error: Unable to read FPS for {video_path}")
        return
    print(f"FPS of video {video_path}: {fps}")
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Finished extracting frames from {video_path}")
            break
        
        # Resize the frame to the desired size
        frame_resized = cv2.resize(frame, (64, 64))
        
        # Save the frame as an image in the output folder
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame_resized)
        
        print(f"Saved frame {frame_count} to {frame_filename}")
        frame_count += 1
    
    cap.release()

# Example usage
video_folder = r'C:\Users\leela\OneDrive\Desktop\DUMB_DEAF\TRAIL'
output_folder = r'C:\Users\leela\OneDrive\Desktop\DUMB_DEAF\DATAFRAMES'

# Ensure the main output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print(f"Main output folder: {output_folder}")

# Iterate through each video file in the folder and extract frames
for video_file in os.listdir(video_folder):
    # Get the full video path
    video_path = os.path.join(video_folder, video_file)
    print(f"Processing video: {video_path}")
    
    # Create a subfolder for each video file based on its name (without extension)
    video_name = os.path.splitext(video_file)[0]  # Remove extension
    video_output_folder = os.path.join(output_folder, video_name)
    
    # Ensure the subfolder exists
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
    print(f"Subfolder for frames: {video_output_folder}")
    
    extract_frames(video_path, video_output_folder)
