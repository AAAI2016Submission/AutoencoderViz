#! /bin/sh

# use the brown dataset and prepare a dataset in bag of words format for the sample vocab

if [ ! -f sample_data/enwik9.zip ]; then
    echo "Wiki data not yet downloaded..."
    echo "Downloading enwik9 file from http://mattmahoney.net/dc/enwik9.zip"
    wget -P sample_data/ http://mattmahoney.net/dc/enwik9.zip
else
    echo "Wiki data zip file already exists. Not downloading."
fi

echo "Unzipping download."
unzip sample_data/enwik9.zip -d sample_data/
echo "Cleaning download."
perl sample_data/wikifil.pl sample_data/enwik9 > sample_data/enwik9cleaneddocs
echo "Running prepareDataset.py script..."
python prepareDataset.py --input sample_data/brown_all_docs_cleaned.txt --outputdir processed_data/sample/ --processor bow --outputformat text --vocab vocabs/sample.txt --dense --normalize

