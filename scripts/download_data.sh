#!/bin/bash
set -e
git clone https://github.com/salesforce/WikiSQL
cd WikiSQL
tar xvjf data.tar.bz2
cd ..
echo "WikiSQL repository cloned and data extracted successfully!"
