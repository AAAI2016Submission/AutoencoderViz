#! /bin/sh

# wiki data 
rm -rf sample_data/enwik9cleandeddocs

# process dataset
rm -rf processed_data/sample

# training files
rm -f models/sample/models/*
rm -f contribs/sample/contribs/*
rm -f hidden_activations/sample/activations/*
rm -f features/sample/features/*

# word cloud images
rm -f visualization/images/sample/*

# word cloud video
rm -f visualization/videos/sample.*

