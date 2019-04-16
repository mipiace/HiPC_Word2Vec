#!/bin/bash

cd data

./getText8.sh
./getBillion.sh

wget https://www.dropbox.com/s/5sdqv854ioody6i/blog_catalog_random_walks
wget https://www.dropbox.com/s/efh8e6wyhu5n3so/ASTRO_PH_random_walks

cd ..