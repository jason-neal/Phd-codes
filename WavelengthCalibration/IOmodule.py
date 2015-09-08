import string

def read_col(filename):
# This program reads column formatted data from a file and
# returns a list in which each sublist correspond to the line's elements.
# THE RESULT IS A LIST OF STRINGS!

	f = open(filename, "r")

	list_data = [];
	
	while 1: 
        	line = f.readline()
        
		if line == "":
                	break
		if line[0] == '#':
                	continue 
        
		line = string.split(line);
		list_data.append(line);
		
	f.close();
	
	return list_data
	
def read_2col(filename):
# The same as the previous, but returns 2 vectors, corresponding each 
# one to a column.THE RESULTS ARE FLOAT PYTHON VECTORS.
# Note that in python all "float" are in fact "double-precision".

	list_data = read_col(filename);
	
	col1 = [];
	col2 = [];
	
	for i in range(len(list_data)):
		#checking if the line is valid
		if(list_data[i][0][0]!='#'):
			col1.append(float(list_data[i][0]));
			col2.append(float(list_data[i][1]));
	
	return [col1, col2];
	
def read_3col(filename):
# The same as the previous, but returns 3 columns

	list_data = read_col(filename);
	
	col1 = [];
	col2 = [];
	col3 = [];
	
	for i in range(len(list_data)):
		#checking if the line is valid
		if(list_data[i][0][0]!='#'):
			col1.append(float(list_data[i][0]));
			col2.append(float(list_data[i][1]));
			col3.append(float(list_data[i][2]));
	
	return [col1, col2, col3];

def read_4col(filename):
# The same as the previous, but returns 4 columns

	list_data = read_col(filename);
	
	col1 = [];
	col2 = [];
	col3 = [];
	col4 = [];
	
	for i in range(len(list_data)):
		#checking if the line is valid
		if(list_data[i][0][0]!='#'):
			col1.append(float(list_data[i][0]));
			col2.append(float(list_data[i][1]));
			col3.append(float(list_data[i][2]));
			col4.append(float(list_data[i][3]));
	
	return [col1, col2, col3, col4];

def read_5col(filename):
# The same as the previous, but returns 5 columns

	list_data = read_col(filename);
	
	col1 = [];
	col2 = [];
	col3 = [];
	col4 = [];
	col5 = [];
	
	for i in range(len(list_data)):
		#checking if the line is valid
		if(list_data[i][0][0]!='#'):
			col1.append(float(list_data[i][0]));
			col2.append(float(list_data[i][1]));
			col3.append(float(list_data[i][2]));
			col4.append(float(list_data[i][3]));
			col5.append(float(list_data[i][4]));
			
	return [col1, col2, col3, col4, col5];	

def read_6col(filename):
# The same as the previous, but returns 6 columns

	list_data = read_col(filename);
	
	col1 = [];
	col2 = [];
	col3 = [];
	col4 = [];
	col5 = [];
	col6 = [];
	
	for i in range(len(list_data)):
		#checking if the line is valid
		if(list_data[i][0][0]!='#'):
			col1.append(float(list_data[i][0]));
			col2.append(float(list_data[i][1]));
			col3.append(float(list_data[i][2]));
			col4.append(float(list_data[i][3]));
			col5.append(float(list_data[i][4]));
			col6.append(float(list_data[i][5]));
			
	return [col1, col2, col3, col4, col5, col6];	

def read_8col(filename):
# The same as the previous, but returns 8 columns

	list_data = read_col(filename);
	
	col1 = [];
	col2 = [];
	col3 = [];
	col4 = [];
	col5 = [];
	col6 = [];
	col7 = [];
	col8 = [];
	
	for i in range(len(list_data)):
		#checking if the line is valid
		if(list_data[i][0][0]!='#'):
			col1.append(float(list_data[i][0]));
			col2.append(float(list_data[i][1]));
			col3.append(float(list_data[i][2]));
			col4.append(float(list_data[i][3]));
			col5.append(float(list_data[i][4]));
			col6.append(float(list_data[i][5]));
			col7.append(float(list_data[i][6]));
			col8.append(float(list_data[i][7]));
			
	return [col1, col2, col3, col4, col5, col6, col7, col8];

def read_9col(filename):
# The same as the previous, but returns 9 columns

	list_data = read_col(filename);
	
	col1 = [];
	col2 = [];
	col3 = [];
	col4 = [];
	col5 = [];
	col6 = [];
	col7 = [];
	col8 = [];
	col9 = [];
	
	for i in range(len(list_data)):
		#checking if the line is valid
		if(list_data[i][0][0]!='#'):
			col1.append(float(list_data[i][0]));
			col2.append(float(list_data[i][1]));
			col3.append(float(list_data[i][2]));
			col4.append(float(list_data[i][3]));
			col5.append(float(list_data[i][4]));
			col6.append(float(list_data[i][5]));
			col7.append(float(list_data[i][6]));
			col8.append(float(list_data[i][7]));
			col9.append(float(list_data[i][8]));
			
	return [col1, col2, col3, col4, col5, col6, col7, col8, col9];
	
def read_10col(filename):
# The same as the previous, but returns 10 columns

	list_data = read_col(filename);
	
	col1 = [];
	col2 = [];
	col3 = [];
	col4 = [];
	col5 = [];
	col6 = [];
	col7 = [];
	col8 = [];
	col9 = [];
	col10 = [];
	
	for i in range(len(list_data)):
		#checking if the line is valid
		if(list_data[i][0][0]!='#'):
			col1.append(float(list_data[i][0]));
			col2.append(float(list_data[i][1]));
			col3.append(float(list_data[i][2]));
			col4.append(float(list_data[i][3]));
			col5.append(float(list_data[i][4]));
			col6.append(float(list_data[i][5]));
			col7.append(float(list_data[i][6]));
			col8.append(float(list_data[i][7]));
			col9.append(float(list_data[i][8]));
			col10.append(float(list_data[i][9]));
			
	return [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10];

def read_12col(filename):
# The same as the previous, but returns 12 columns

	list_data = read_col(filename);
	
	col1 = [];
	col2 = [];
	col3 = [];
	col4 = [];
	col5 = [];
	col6 = [];
	col7 = [];
	col8 = [];
	col9 = [];
	col10 = [];
	col11 = [];
	col12 = [];
	
	for i in range(len(list_data)):
		#checking if the line is valid
		if(list_data[i][0][0]!='#'):
			col1.append(float(list_data[i][0]));
			col2.append(float(list_data[i][1]));
			col3.append(float(list_data[i][2]));
			col4.append(float(list_data[i][3]));
			col5.append(float(list_data[i][4]));
			col6.append(float(list_data[i][5]));
			col7.append(float(list_data[i][6]));
			col8.append(float(list_data[i][7]));
			col9.append(float(list_data[i][8]));
			col10.append(float(list_data[i][9]));
			col11.append(float(list_data[i][10]));
			col12.append(float(list_data[i][11]));
			
	return [col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12];


def write_2col(filename, data1, data2):
# Writes data in 2 columns separated by tabs in a "filename" file.

	f = open(filename, "w")
	
	for i in range(len(data1)):
		f.write("\t"+str(data1[i])+"\t\t"+str(data2[i])+"\n")
	
	f.close();
		
