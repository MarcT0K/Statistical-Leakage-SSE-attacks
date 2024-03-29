#!/bin/bash
set -e

sudo apt install python3 python3-pip wget unzip

pip3 install -r requirements.txt

# Enron Dataset
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz -O enron.tar.gz
tar xzvf enron.tar.gz
rm enron.tar.gz

# Apache Dataset
mkdir apache_ml
cd apache_ml
for y in {2002..2011}; do
    for m in {01..12}; do
        wget "http://mail-archives.apache.org/mod_mbox/lucene-java-user/${y}${m}.mbox"
    done
done

# Blogs
wget https://web.archive.org/web/20200121222642/http://u.cs.biu.ac.il/~koppel/blogs/blogs.zip
unzip blogs.zip
rm blogs.zip

# Dependencies for the figures
sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
