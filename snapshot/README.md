# Capturing images (Two Options)

## Basic Operations
1. "Space" - Capture the current frame
    * capture and label about 2-3 frames per vehicle
2. "Esc" - Skip / exit the current video
3. Key "P" - Pause video
    * Press "P" again to continus
4. Key "D" - Skip 10 frames (useful when only a few vehicles are in the video)
    * Note: cann't go back to previous frames
    
## Option 1: Capture images from a specific video
1. Set the input, video, and output directory, video_name in snapshot.py
    * For example, for video: 20191210-0733_CAM2_0073.MP4:
    ```
    input_dir = "/Volumes/VERBATIM HD/5805_DOT_ArterialNetworkFootage/Site01-StanleyAve,MountWaverley/Camera01/C341/DCIM/101MEDIA"
    video_dir = "Site01-StanleyAve,MountWaverley"
    video_name = "20191210-0733_CAM2_0073.MP4"
    output_dir = "images/"
    ```

2. Run snapshot.py in terminal:
    ```
    python snapshot.py
    ```

3. This will play the specific video in input_dir, and save the images to output_dir/video_dir/
## Option 2: Capture images from all videos
1. Set the input, video, and output directory, video_name in snapshot_playall.py:
    ```
    input_dir = "/Volumes/VERBATIM HD/5805_DOT_ArterialNetworkFootage/Site01-StanleyAve,MountWaverley/Camera01/C341/DCIM/101MEDIA"
    video_dir = "Site01-StanleyAve,MountWaverley"
    output_dir = "images/"
    ```
2. Run `snapshot_playall.py` in terminal:
    ```
    python snapshot_playall.py
    ```
3. This will play all the video in input_dir, and save the images to output_dir/video_dir/
4. Hint: You can specify a start point of the playlist
    * For example, to play the videos starting from 20191210-0203_CAM2_0040.MP4, run: (Sorted by video filenames in ascending order)
    ```
    python snapshot_playall.py -s 20191210-0203_CAM2_0040.MP4

    ```
    ("-s" is the optional argument for a start point of playlist)
5. Hint2: Press Ctrl+C in terminal if you want to exit all the videos (exit the program)



