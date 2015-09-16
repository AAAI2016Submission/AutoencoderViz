#! /bin/sh

# $1 = path
# $2 = framerate
# $3 = output file name
avconv -f image2 -i $1/%08d.png -r $2 $3
