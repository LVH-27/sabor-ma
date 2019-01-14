#!/bin/bash

thresh=$2
for f in $1/*
do
	file_n=$(echo $f | cut -f2 -d- | cut -f1 -d.)
	if [[ $file_n -lt $thresh ]]
	then
		rm $f
	fi
done
