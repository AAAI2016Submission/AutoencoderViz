#! /bin/sh


# first generate the full video
sh visualization/genVideoFFmpeg.sh visualization/images/sample 30 visualization/videos/sample/full.mp4

# now create timelapse
sh visualization/genTimeLapseFFmpeg.sh visualization/videos/sample/full.mp4 .05 visualization/videos/sample/timelapse.mp4
