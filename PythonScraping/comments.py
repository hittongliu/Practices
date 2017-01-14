#-*- coding: utf-8 -*-
from download import *
import re
import sys
import chardet
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
reload(sys)
sys.setdefaultencoding('utf-8')
def craw(url):
    html = download(url)
    soup = BeautifulSoup(html, 'html.parser')
    uls = soup.find_all('div', attrs = {'class':'J_brief-cont'})
    restr = 'user-info">[\s\S]+?irr-star(\d)0"'
    scores = re.findall(restr,html)
    print scores
    print 'comments', len(uls)
    print 'scores', len(scores)
    if len(uls) != len(scores):return
    comments = []
    negfile = open('./neg.txt','a', 0)
    posfile = open('./pos.txt','a', 0)
    i = 0
    for ul in uls:
        stru = (ul.text.strip())
        if int(scores[i]) >= 4:
            posfile.write(stru)
            posfile.write('\n')
        if int(scores[i]) < 4:
            negfile.write(stru)
            negfile.write('\n')
        i += 1
    negfile.close()
    posfile.close()
listmember = []
file = open('./shope1')
for linefile in file.readlines():
    data = linefile.strip().split()
    listmember.append(map(int, data))
for shop in listmember:
    shopurl = 'http://www.dianping.com/shop/%s/' % str(shop[0])
    for j in range(5):
        star = 'review_more_%sstar' % str(j + 1)
        url = shopurl + star
        craw(url)
        time.sleep(j * 3)
    time.sleep(5)
