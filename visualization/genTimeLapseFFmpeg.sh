#! /bin/sh

# $1 input video file
# $2 decimal portion to keep
# $3 output file name

ffmpeg -i $1 -r 8 -filter:v "setpts="$2"*PTS" -an $3
