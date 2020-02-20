import cv2
import time
import os
import argparse

# Set input and output directories
input_dir = "Video/Site01-StanleyAve,MountWaverley"
video_dir = "Site01-StanleyAve,MountWaverley"
output_dir = "images/"

def snapshot(video_name):
    
    input_video = os.path.join(input_dir, video_name)

    # Loading video
    vidcap = cv2.VideoCapture(input_video)

    img_count = 0
    
    pause_flag = False

    while vidcap.isOpened():
        
        # read frame if video not paused
        if not pause_flag:
            # read frame and show in window
            success, image = vidcap.read()
            if not success:
                break
            cv2.imshow("Truck_Video", image)

        # Key for snapshot or exit
        key = cv2.waitKey(1)
        if key%256 == 27:
            # ESC pressed
            print("Escape hit, closing..." + video_name)
            break
        elif key%256 == 32:
            # SPACE pressed
            img_name = os.path.join(output_dir, video_dir, str(video_name[:-4]) + "_" 
                                                         + str(img_count).zfill(4) 
                                                         + ".jpg")
            cv2.imwrite(img_name, image)
            print("{} written!".format(img_name))
            img_count += 1
        elif key%256 == 112:
            # Key "P" pressed
            pause_flag = not pause_flag
        elif key%256 == 100:
            # Key "D" pressed
            for i in range(20):
                vidcap.grab()
    
    vidcap.release()
    

def get_args():
    """ Obtaining args from terminal """
    parser = argparse.ArgumentParser(description="Snapshot")
    # filenames
    parser.add_argument("-s", "--start_from", type = str, default = None, 
                        required = False, help = "Twitter data file")
    args = parser.parse_args()
    
    return args

def main():
    
    # read optional argument:start_from
    args = get_args()

    # Window for snapshots
    cv2.namedWindow("Truck_Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Truck_Video", 800, 500)

    # Create output dir if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, video_dir)):
        os.makedirs(os.path.join(output_dir, video_dir))
         
    sorted_videonames = sorted(os.listdir(input_dir))
        
    if args.start_from == None:
        # play all video from the begining
        for video_name in sorted_videonames:

            if video_name.endswith(".MP4") and not video_name.startswith("._"):
		# if not os.path.exists(os.path.join(output_dir, video_dir, video_name)):
		#    os.makedirs(os.path.join(output_dir, video_dir, video_name))
                snapshot(video_name)
    else:
        # play all video from "start_from"
        
        index = sorted_videonames.index(args.start_from)
        
        for video_name in sorted_videonames[index:]:
            if video_name.endswith(".MP4") and not video_name.startswith("._"):
                snapshot(video_name)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()