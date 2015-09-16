#! /bin/sh

# use the brown dataset and prepare a dataset in bag of words format for the sample vocab

echo "Running prepareDataset.py script..."
python prepareDataset.py --input sample_data/brown_all_docs_cleaned.txt --outputdir processed_data/sample/ --processor bow --outputformat text --vocab vocabs/sample.txt --dense --normalize

