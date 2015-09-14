#!/usr/bin/env python
# -*- coding: utf8 -*-

# Get Filenames of pedros codes

import fnmatch
import os

# change back
#os.chdir(old_dir) 


def get_filenames(path, regexp, regexp2=False):
	""" regexp must be a regular expression as a string
			eg '*.ms.*', '*_2.*', '*.ms.norm.fits*'

		resexp2 is if want to match two expressions such as 
			'*_1*' and '*.ms.fits*'
	"""
	os.chdir(path)
	filelist = []
	for file in os.listdir('.'):
		if regexp2:  # Match two regular expresions
			if fnmatch.fnmatch(file, regexp) and fnmatch.fnmatch(file, regexp2):
				#print file
				filelist.append(file)
		else:
			if fnmatch.fnmatch(file, regexp):
				#print file
				filelist.append(file)
	filelist.sort()
	return filelist


def main():
	path = "/home/jneal/data/BrownDwarfs-PedrosCode/HD30501-1/"
# #old_dir = os.curdir()
# os.chdir(path)
# filelist = []
# for file in os.listdir('.'):
#     if fnmatch.fnmatch(file, '*1.nod.ms.norm.fits*'):
#         print file
#         filelist.append(file)

# filelist.sort()
# print(filelist)
	list1 = get_filenames(path, "*.ms.*")
	for file in list1:
		pass#print file
	list2 = get_filenames(path, "*.norm.*","*_1.*")
	for file in list2:
		pass#print file


if __name__ == '__main__':
	main()