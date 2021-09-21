import sys
import requests
from bs4 import BeautifulSoup
import os
import time

def crawler(url, post_name):
    last_page = False
    count = 1

    if not os.path.exists(post_name):
                os.makedirs(post_name)

    while not last_page:
        r = requests.get(url) # 將此頁面的HTML GET下來
        soup = BeautifulSoup(r.text,"html.parser") # 將網頁資料以html.parser

        # get image and save image
        img = soup.find_all("img", align="center")
        if not img:
            print("cannot get image url for count = %d" %(count))
        else:
            img_url = img[0]["src"]
            char = img[0]["title"]
            pic = requests.get(img_url)
            img2 = pic.content
            pic_out = open(post_name+'/'+char+ '.png','wb')
            pic_out.write(img2)
            pic_out.close() # 關閉檔案(很重要)

        count += 1

        # get next page's url
        u = soup.find_all("a", text="下一張")
        if not u:
            last_page = True
        else:
            url = "http://163.20.160.14/~word/modules/myalbum/" + u[0]["href"]
            time.sleep(3) # 如果太頻繁的抓，會造成網頁crash，需要等一下下


url = sys.argv[1]
post_name = sys.argv[2]
crawler(url, post_name)
