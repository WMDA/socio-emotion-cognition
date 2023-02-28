#! /bin/bash

cd /data/project/BEACONB/CNSCNSD/scans_t2

file=(*)

for i in ${file[@]:66}; 
do 
	echo $i
	cd $i/DICOM/*/;
	tar -xf *.0005.tar.bz2;
	tar -xf *.0006.tar.bz2;
	tar -xf *.0007.tar.bz2;
	tar -xf *.0008.tar.bz2;
	tar -xf *.0009.tar.bz2;
	cd ../../.. 

done

#2022