import urllib2
import re
import time
import numpy as np

def download(url, user_agent = 'hssp', num_retries = 2):
  print 'Downloading:', url
  headers = {'User-agent':user_agent}
  request = urllib2.Request(url, headers = headers)
  try:
    html = urllib2.urlopen(request).read()
  except urllib2.URLError as e:
    print 'Downloading error:', e.reason
    html = None
    if num_retries > 0:
      if hasattr(e, 'code') and 500 <= e.code <=600:
        return download(url,user_agent, num_retries - 1)
  return html

def crawl_sitemap(url, restr):
  sitemap = download(url)
  links = re.findall(restr, sitemap)
  links = set(links)
  return links
  # for link in links:
  #   print link
  #   print 'link'










