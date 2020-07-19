import subprocess
    
# reset frames folder
subprocess.call([
    "ffmpeg", "-framerate", "50", "-y", "-i", "frames/frame-%010d.png", "-r", "30", "-pix_fmt", "yuv420p", "dumped_video.mp4"
])

