import itertools
import 1.4.1.py

page = 1
linkpage = set()
while True:
  if page < 10:
    strpage = '0' + str(page)
  else:
    strpage = str(page)
  url = 'http://www.dianping.com/search/category/1/10/o10p%s' % strpage
  linkpage = linkpage | crawl_sitemap(url, 'href="/shop/(\d+?)"')
  page += 1
  if page == 51:
    break
  time.sleep(2 + page / 100.0)
linknum = list(linkpage)
# np.save('./member', linknum)
members = (np.load('./member.npy'))
print len(members)