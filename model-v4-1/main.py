#!/usr/bin/env python3
"""
Example usage of the Actor Tracking System
Demonstrates basic and advanced usage patterns
"""

from actor_tracker import ActorTracker
import cv2
import json

def basic_example():
    """Basic usage example"""
    print("🎬 Basic Actor Tracking Example")
    
    # Initialize the tracker
    tracker = ActorTracker(
        model_path='yolov8n.pt',  # Will download automatically if not present
        conf_threshold=0.5,
        device='auto'  # Automatically use CUDA if available
    )
    
    # Process a video file
    results = tracker.process_video(
        video_path='../movies/video_5.mp4.mp4',
        output_path='output/tracked_movie.mp4',
        skip_frames=1,  # Process every frame (set higher for faster processing)
        max_frames=None  # Process entire video
    )
    
    # Print results
    print(f"✅ Detected {results['summary']['total_actors']} unique actors")
    
    # Show top actors by screen time
    for actor_name, data in sorted(results['actors'].items(), 
                                 key=lambda x: x[1]['screen_frames'], reverse=True)[:5]:
        print(f"🎭 {actor_name}: {data['screen_frames']} frames")

def advanced_example():
    """Advanced usage with custom configuration"""
    print("🚀 Advanced Actor Tracking Example")
    
    # Use higher quality model for better accuracy
    tracker = ActorTracker(
        model_path='yolov8x.pt',  # Larger, more accurate model
        conf_threshold=0.3,       # Lower threshold to catch more detections
        device='cuda'             # Force CUDA usage
    )
    
    # Process with custom settings
    results = tracker.process_video(
        video_path='input_movie.mp4',
        output_path='high_quality_tracked.mp4',
        skip_frames=2,            # Skip every other frame for speed
        max_frames=1000           # Process first 1000 frames only
    )
    
    # Analyze results
    analyze_results(results)

def analyze_results(results):
    """Analyze and visualize tracking results"""
    print("\n📊 DETAILED ANALYSIS")
    print("=" * 50)
    
    actors = results['actors']
    
    # Screen time analysis
    total_frames = results['summary']['total_frames_processed']
    print(f"Total frames analyzed: {total_frames}")
    
    for actor_name, data in actors.items():
        screen_percentage = (data['screen_frames'] / total_frames) * 100
        
        print(f"\n🎭 {actor_name}:")
        print(f"   Screen time: {data['screen_frames']} frames ({screen_percentage:.1f}%)")
        print(f"   First appearance: frame {data['first_appearance']}")
        print(f"   Last appearance: frame {data['last_appearance']}")
        print(f"   Average confidence: {data['avg_confidence']:.3f}")
        print(f"   Face detections: {data['face_detections']}")
        
        if data['face_detections'] > 0:
            print(f"   Face recognition confidence: {data['face_confidence']:.3f}")

def batch_processing_example():
    """Example of processing multiple videos"""
    import os
    from pathlib import Path
    
    print("🎬 Batch Processing Example")
    
    # Initialize tracker once for multiple videos
    tracker = ActorTracker(
        model_path='yolov8s.pt',
        conf_threshold=0.4
    )
    
    # Process all videos in a directory
    video_dir = Path("input_videos")
    output_dir = Path("tracked_videos")
    output_dir.mkdir(exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    for video_file in video_dir.iterdir():
        if video_file.suffix.lower() in video_extensions:
            print(f"Processing: {video_file.name}")
            
            output_path = output_dir / f"tracked_{video_file.name}"
            
            try:
                results = tracker.process_video(
                    video_path=str(video_file),
                    output_path=str(output_path),
                    skip_frames=3,  # Faster processing for batch
                    max_frames=5000  # Limit frames for demo
                )
                
                print(f"✅ Completed: {video_file.name}")
                print(f"   Actors found: {results['summary']['total_actors']}")
                
            except Exception as e:
                print(f"❌ Error processing {video_file.name}: {e}")

def test_with_webcam():
    """Test tracking with webcam (real-time demo)"""
    print("📹 Webcam Demo")
    
    tracker = ActorTracker(
        model_path='yolov8n.pt',  # Fast model for real-time
        conf_threshold=0.5
    )
    
    cap = cv2.VideoCapture(0)  # Use webcam
    frame_count = 0
    
    print("Press 'q' to quit, 'r' to reset tracking")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame (simplified for real-time)
            tracks = tracker.process_frame(frame, frame_count)
            
            # Draw results
            annotated_frame = tracker.draw_results(frame, tracks)
            
            # Display
            cv2.imshow('Actor Tracking Demo', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset tracker
                tracker.tracker = tracker.tracker.__class__()
                print("Tracking reset")
            
            frame_count += 1
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run the examples
    # import sys
    basic_example()    
    # if len(sys.argv) > 1:
        # example_type = sys.argv[1]
        #
        # if example_type == "basic":
        #     basic_example()
        # elif example_type == "advanced":
        #     advanced_example()
        # elif example_type == "batch":
        #     batch_processing_example()
        # elif example_type == "webcam":
        #     test_with_webcam()
        # else:
        #     print("Available examples: basic, advanced, batch, webcam")
    else:
        print("🎬 Actor Tracking System Examples")
        print("Usage: python example_usage.py [basic|advanced|batch|webcam]")
        print("\nRunning basic example...")
        basic_example()
