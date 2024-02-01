# from lxml import etree
# import requests
#
#
# # 要爬取的url，注意：在开发者工具中，这个url指的是第一个url
# url = "https://www.bilibili.com/v/popular/rank/all"
#
# # 模仿浏览器的headers
# headers = {
#     "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# }
# params = {
#     "mid": 387694560,
#     "pn": 1,
#     "ps": 25,
#     "index": 1,
#     "jsonp": "jsonp"
# }
# # get请求，传入参数，返回结果集
# resp = requests.get(url,params=params)
# print(resp.json())
# # 将结果集的文本转化为树的结构
# tree = etree.HTML(resp.text)
#
# # 定义列表存储所有数据
# dli = []
# # 遍历所有数据
# for s in range(1,101):
#     li = []
#     #根据树的路径找到对应的数据集
#     num = tree.xpath("/html/body/div[3]/div[2]/div[2]/ul/li["+str(s)+"]/div[1]/text()")  # 获取热搜排序
#     name = tree.xpath("/html/body/div[3]/div[2]/div[2]/ul/li["+str(s)+"]/div[2]/div[2]/a/text()")# 获取标题
#     url = tree.xpath("/html/body/div[3]/div[2]/div[2]/ul/li["+str(s)+"]/div[2]/div[2]/a/@href")#获取链接
#     look = tree.xpath("/html/body/div[3]/div[2]/div[2]/ul/li["+str(s)+"]/div[2]/div[2]/div[1]/span[1]/text()")# 获取播放量
#     say = tree.xpath("/html/body/div[3]/div[2]/div[2]/ul/li["+str(s)+"]/div[2]/div[2]/div[1]/span[2]/text()") # 获取评论量
#     up = tree.xpath("/html/body/div[3]/div[2]/div[2]/ul/li["+str(s)+"]/div[2]/div[2]/div[1]/a/span/text()") # 获取up主
#     score = tree.xpath("/html/body/div[3]/div[2]/div[2]/ul/li["+str(s)+"]/div[2]/div[2]/div[2]/div/text()") # 获取综合得分
#     #获取数据集中的元素
#     li.append(num[0])
#     li.append(name[0])
#     li.append(url[0])
#     li.append(look[0])
#     li.append(say[0])
#     li.append(up[0])
#     li.append(score[0])
#
#     dli.append(li)
# # 打印数据
# for dd in dli:
#     print(dd)

import requests
from lxml import html
import pandas as pd
# url = 'https://movie.douban.com/top250?start=0&filter='

headers = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
# response = requests.get(url,headers = headers)
# print(response)
# url='https://movie.douban.com/' #需要爬数据的网址

url='https://www.qtccolor.com/secaiku/color/106462' #需要爬数据的网址
page=requests.Session().get(url,headers = headers)
tree=html.fromstring(page.text)
# result=tree.xpath('//td[@class="title"]//a/text()') #获取需要的数据
result=tree.xpath('//li[@class="answer"]/text()') #获取需要的数据
print(result[1])
sp=result[1][10:14]
print(sp)
rgb=result[1].split("，")[0].split("(")[1].split(")")[0].split(",")
print(rgb)
rgb.append(sp)
print(rgb)
# print(result[1][27:30])
# print(result[1][31:34])
# print(result[1][35:38])



url='https://www.qtccolor.com/secaiku/dir/22' #需要爬数据的网址
page=requests.Session().get(url,headers = headers)
tree=html.fromstring(page.text)
# result=tree.xpath('//td[@class="title"]//a/text()') #获取需要的数据
urls=tree.xpath('//div[@class=" pc-width"]//a/@href') #获取需要的数据
print(len(urls))
base_path = "https://www.qtccolor.com//"
all_rgb=[]
for re in urls:
    path=base_path+re
    page = requests.Session().get(path, headers = headers)
    tree = html.fromstring(page.text)
    result = tree.xpath('//li[@class="answer"]/text()')  # 获取需要的数据
    print(result[1])
    sp = result[1][10:14]
    # print(sp)
    rgb = result[1].split("，")[0].split("(")[1].split(")")[0].split(",")
    # print(rgb)
    rgb.append(sp)
    all_rgb.append(rgb)
rgb_d=pd.DataFrame(all_rgb)
rgb_d.to_csv("panton.csv")