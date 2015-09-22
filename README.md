# AutoencoderViz
An autoencoder built specifically to work with and understand how networks learn on natual language data.

## Project dependencies:
  * [Lua Torch](torch.ch/docs/getting-started.html)
  * Python3
  * Numpy + Scipy for python3 (best installed using pip3)
  * SciKitLearn for python3 (best installed using pip3)
  * Matplotlib for python3 (best installed using pip3)
  * [Python word cloud generator ](https://github.com/amueller/word_cloud)
  * FFmpeg or avconv (part of libav)

## Example usage:

 1. Preprocess data
   * Run `sh samplePreprocess.sh` script
   * This will process the brown corpus and prepare it for training. The output will be a normalized tf-idf matrix representing bag-of-words values. The words represented in each bag-of-words vector will be restricted to the words in the supplied vocab file. 
 2. Create model format file
   * There is an existing model_format file for this example in the `model_formats/sample/` directory. This file defines the input and output to the network, how often to save data, where to save data, and the structure of the network including the number of nodes in the input and hidden layers and transfer functions.
 3. Train network
   * Run `sh sampleTrain.sh`
   * This will start the training process and will run 10000 iterations on one layer. It will save the model and the max activations every single iteration and the hidden features every 10 iterations.
 4. Visualize with TSNE or nearest neighbors
   * Run `sh sampleVizTSNE.sh` to look at a plot of the TSNE dimensionality reduction.
   * Run `sh sampleAnalysisNN.sh` to experiment with nearest neighbors.
 5. Generate word cloud images for contribs
   * To visually examine what each node has learned you'll have to generate the word cloud images.
   * Run `sh sampleGenerateWordClouds.sh`.
   * You can look at some of those images. There is one for each iterations, 10000 total. They should be in the visualization/images/sample/ directory.
 6. Generate video from word cloud images.
   * Now using the images generated in the previous step as frames, generate the video.
   * If you have FFmpeg installed try: `sh sampleGenerateVideoFFMPEG.sh`
   * If you have avconv installed try: `sh sampleGenerateVideoAvconv.sh`
   * This step may take a few minutes.
   * One long, normal-speed video should be generated and saved to visualization/videos/sample_full.[avi|mp4] and one time lapse should be saved to visualization/videos/sample_timelapse.[avi|mp4].
 
## Sample results:

 * The following three images are from a run with a small vocab of only 41 words each in one of 4 categories: basketball, football, days, months.
  * After a single iteration of training:
    ![1itrs](https://raw.githubusercontent.com/AAAI2016Submission/AutoencoderViz/master/visualization/images/dataset_wiki_vocabdaysmonthsbasketballfootball_gray2/00000001.png "The network after a single iteration of training")
  * After the first layer is fully trained:
    ![20000itrs](https://raw.githubusercontent.com/AAAI2016Submission/AutoencoderViz/master/visualization/images/dataset_wiki_vocabdaysmonthsbasketballfootball_gray2/00020000.png "After the first layer is fully trained")
  * After both the first and second layers are fully trained:
    ![35000itrs](https://raw.githubusercontent.com/AAAI2016Submission/AutoencoderViz/master/visualization/images/dataset_wiki_vocabdaysmonthsbasketballfootball_gray2/00035000.png "After fully training two layers")
