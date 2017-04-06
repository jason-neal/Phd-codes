
# coding: utf-8

# In[5]:

## Team arxiv look-up 
import pandas as pd
import numpy as np
from __future__ import print_function
# arxiv api?


# In[6]:

# load list of peoples names 

# for each name on list search peoples arxiv listing


# append keywords of papers to a large list


#keyword_list = []
#keyword_list.append()


# In[12]:

def print_article_keywords(keywords, limit_value=5):
    keywords = [word.lower() for word in keywords]

    #keyword_counts = {key:keywords.count(key) for key in np.unique(keywords)}
    #x = [[val, key] for key, val in keyword_counts.items()]
    x = [[keywords.count(key), key] for key in np.unique(keywords)]
    x.sort()
    x=x[::-1]   # reverse list  (there should be a sort option for this)
    #print("x sorted", x)
    
    print("Research Strengths:" )
    [print("\t " + pair[-1].capitalize()) for num, pair in enumerate(x) if num < limit_value]
    
    


# In[13]:

# Some testing values before I find an arxiv api
# (I know there is one as I was using it for arxivTTS)
testing=True
test_keywords = ['phd', 'pHd', 'phD', 'ice cream', "pokemon", "pokemon", "pokemon", 'solar system', 'Physics', "dogs", 'Phd', 'PhD','stars','solar system', 'planets', 'spectroscopy', 'stars', 'stars', 'Planets', "parameters", "stars", 'transit','transit']

Researcher_list = ["A. Name", "J. J. Neal"]


# In[14]:

#Researcher_list = np.loadtxt()?
for name in Researcher_list:
    print("Looking for arxiv atricles for: {0}".format(name))
    keyword_list = []
    #keyword_list.append()
    #articles = arxiv lookup grab keywords
    articles = ""
    for article in articles:
        #article_keywords = article.keywords
        #keyword_list.append(article_keywords)
        #keyword_list = keywordlist + article_keywords   # or this
        pass
    if testing: 
        keyword_list = keyword_list + test_keywords
    #print(keyword_list)
    print("Researcher:", name)
    print_article_keywords(keyword_list, limit_value=7)


# In[ ]:




# In[16]:

print("Researcher:", name)
print_article_keywords(test_keywords, limit_value=6)


# In[17]:

# Create big string list to write to file.


# In[18]:

filename = "Researcher_strengths.txt"
with open(filename, "w+") as f:
#%np.savetxt()
    f.write("Test save")


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# # Play with arxiv API

# In[22]:

import urllib
url = 'http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1'
data = urllib.request.urlopen(url)

print(data)


# In[23]:

import requests
url = 'http://export.arxiv.org/api/query?search_query=au:del_maestro&start=0&max_results=1'
url2 = 'http://export.arxiv.org/api/query?search_query=au:Santos&start=0&max_results=5'


# In[24]:

request_data = requests.get(url2)


# In[25]:

print(request_data)


# In[26]:

#request_data.text


# In[27]:

request_data.headers['content-type']


# Arxiv doesn"t have keywords on the articles. am I able to get the bib reference for it maybe and look in there?

# In[ ]:




# # search for article I found in adsabs
# http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1607.03906
# #bib entry for this
#  http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2016arXiv160703906S&data_type=BIBTEX&db_key=PRE&nocookieset=1
#         
#         
#         # another
#         
#         # arxiv
#         https://arxiv.org/abs/1607.03684
#             # ads
#         http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1607.03684
#    # to find bib
# http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2016arXiv160703684B&data_type=BIBTEX&db_key=PRE&nocookieset=1
#                 

# In[ ]:




# In[29]:

p = requests.get("https://arxiv.org/abs/1607.03684")
q = requests.get("http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1607.03684")
r = requests.get("http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:1607.03684")


# In[38]:

print(p)
print(q)
print(r)

print(r.headers['content-type'])
print(r.encoding)
#print(r.text)
print(r.json)


# In[31]:

s = requests.get("http://adsabs.harvard.edu/cgi-bin/nph-abs_connect?db_key=AST&db_key=PRE&qform=AST&arxiv_sel=astro-ph&arxiv_sel=cond-mat&arxiv_sel=cs&arxiv_sel=gr-qc&arxiv_sel=hep-ex&arxiv_sel=hep-lat&arxiv_sel=hep-ph&arxiv_sel=hep-th&arxiv_sel=math&arxiv_sel=math-ph&arxiv_sel=nlin&arxiv_sel=nucl-ex&arxiv_sel=nucl-th&arxiv_sel=physics&arxiv_sel=quant-ph&arxiv_sel=q-bio&sim_query=YES&ned_query=YES&adsobj_query=YES&aut_logic=OR&obj_logic=OR&author=Neal%2C+Jason+J&object=&start_mon=&start_year=&end_mon=&end_year=&ttl_logic=OR&title=&txt_logic=OR&text=&nr_to_return=200&start_nr=1&jou_pick=ALL&ref_stems=&data_and=ALL&group_and=ALL&start_entry_day=&start_entry_mon=&start_entry_year=&end_entry_day=&end_entry_mon=&end_entry_year=&min_score=&sort=SCORE&data_type=SHORT&aut_syn=YES&ttl_syn=YES&txt_syn=YES&aut_wt=1.0&obj_wt=1.0&ttl_wt=0.3&txt_wt=3.0&aut_wgt=YES&obj_wgt=YES&ttl_wgt=YES&txt_wgt=YES&ttl_sco=YES&txt_sco=YES&version=1")



# In[ ]:

s.text
print(s.headers)

print("done")


# In[ ]:




# In[ ]:

first_name = "Jason"
last_name = "Neal"
middle_inital = "J" 
exo_name_request = "http://adsabs.harvard.edu/cgi-bin/nph-abs_connect?db_key=AST&db_key=PRE&qform=AST&arxiv_sel=astro-ph&arxiv_sel=cond-mat&arxiv_sel=cs&arxiv_sel=gr-qc&arxiv_sel=hep-ex&arxiv_sel=hep-lat&arxiv_sel=hep-ph&arxiv_sel=hep-th&arxiv_sel=math&arxiv_sel=math-ph&arxiv_sel=nlin&arxiv_sel=nucl-ex&arxiv_sel=nucl-th&arxiv_sel=physics&arxiv_sel=quant-ph&arxiv_sel=q-bio&sim_query=YES&ned_query=YES&adsobj_query=YES&aut_logic=OR&obj_logic=OR&author={0}%2C+{1}+{2}&object=&start_mon=&start_year=&end_mon=&end_year=&ttl_logic=OR&title=&txt_logic=OR&text=&nr_to_return=200&start_nr=1&jou_pick=ALL&ref_stems=&data_and=ALL&group_and=ALL&start_entry_day=&start_entry_mon=&start_entry_year=&end_entry_day=&end_entry_mon=&end_entry_year=&min_score=&sort=SCORE&data_type=SHORT&aut_syn=YES&ttl_syn=YES&txt_syn=YES&aut_wt=1.0&obj_wt=1.0&ttl_wt=0.3&txt_wt=3.0&aut_wgt=YES&obj_wgt=YES&ttl_wgt=YES&txt_wgt=YES&ttl_sco=YES&txt_sco=YES&version=1".format(last_name, first_name, middle_inital)
exo_request = requests.get(exo_name_request)
#print(exo_request.text)
print(exo_request.headers)
#print(exo_request.text)

print("done")


# In[ ]:



