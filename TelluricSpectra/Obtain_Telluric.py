#!/usr/bin/env python
# -*- coding: utf8 -*-


## Module that has a function that can obtain the telluric spectra relevant to the observations we have
 ## ie. match date, time, RA/DEC

from Get_filenames import get_filenames



def Get_Telluric(date, time):
	"""Find telluric spectra that matches the input conditions of the obersvation """
	
	tapas_time = str(int(time[0:2]) + 1) + time[2:]
	print("date :",date)
	print()
	#timing error = 
	string1 = "tapas_"  #+ date
	print("string1",string1)
	string2 = tapas_time
	print("string2",string2)
	matching = get_filenames(Tapas_path, string1 , string2)
	matching_date = get_filenames(Tapas_path, string1)
	return matching, matching_date
	#pass





if __name__== "__main__" :
	Tapas_path = "/home/jneal/Phd/data/Tapas/"
	#tapas_2012-08-02T10:01:44-2452
	print("Begining Telluric request")
	Test_target = "HD30501-1"
	Test_date = "2012-04-07" # date yy-mm-dd
	Test_time = "00:20:20"  # hh:mm:ss
	Test_RA =	"04:45:38"
	Test_DEC = "-50:04:38"

	Test_Result, Resultdate = Get_Telluric(Test_date, Test_time)
	print("TEST Result", Test_Result)
	print("TEST Resultdate", Resultdate)