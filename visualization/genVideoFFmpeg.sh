#! /bin/sh

# $1 = path to images
# $2 = framerate
# $3 = output file name

ffmpeg -framerate $2 -i $1/%08d.png -c:v libx264 -r 30 -pix_fmt yuv420p $3
