#! /bin/bash


sphinx-apidoc -f -o source/ ..

sphinx-apidoc -f -o source/ ../tests

make html
