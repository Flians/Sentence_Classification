# -*- coding: utf-8 -*-
import os
from urllib import request
from urllib import parse
from urllib import error
from http import cookiejar
from bs4 import BeautifulSoup
import re
from PIL import Image
import time
import socket
import random
import csv
import jieba
import json

# 用于保存节点信息
checkpoint = {
    "id_moive":0,
    "id_page":0,
    "id_item":0,
    "url_moive":""
}

# 读取停用词
# stopwords = [line.strip() for line in open('./stopwords.txt', encoding='utf-8').readlines()]

# 登录页面信息
login_url = 'https://accounts.douban.com/j/mobile/login/basic'
form_data = {
    "ck":'',
    "name":"",
    "password":'',
    'remember':'false',
    "ticket":''
}

user_agent_list = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
    "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
    "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
    "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11"
]
user_agent = random.choice(user_agent_list)

user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'

headers = {'User-Agnet': user_agent, 'Connection': 'keep-alive'}

# 保存cookies便于后续页面的保持登陆
cookie = cookiejar.CookieJar()
cookie_support = request.HTTPCookieProcessor(cookie)
opener = request.build_opener(cookie_support)

# 用于生成短评页面网址的函数
def MakeUrl(mian_url,start):
    """make the next page's url"""
    url = mian_url + '/comments?start=' + str(start) + '&limit=20&sort=new_score&status=P'
    print(url)
    return url

# 编码信息，生成请求，打开页面获取内容
def do_req(url, form_data=None):
    req = request.Request(url=url, headers=headers)
    if form_data:
        req.data = parse.urlencode(form_data).encode('utf-8')
    res = opener.open(req).read().decode('utf-8')
    return res

def login():
    # 登录
    res_login = do_req(login_url, form_data)

    # 获取验证码图片地址
    soup = BeautifulSoup(res_login, "html.parser")
    if soup.find('img', id='captcha_image'):
        captchaAddr = soup.find('img', id='captcha_image')['src']

        # 匹配验证码id
        reCaptchaID = r'<input type="hidden" name="captcha-id" value="(.*?)"/'
        captchaID = re.findall(reCaptchaID, res_login)

        # 下载验证码图片
        request.urlretrieve(captchaAddr, "captcha.jpg")
        # 显示验证码
        Image.open('captcha.jpg').show()
        # 输入验证码并加入提交信息中，重新编码提交获得页面内容
        captcha = input('please input the captcha:')
        form_data['captcha-solution'] = captcha
        form_data['captcha-id'] = captchaID[0]

        # 带验证码登录
        res_login = do_req(login_url, form_data)

def get_movies_comments(mian_url,id_page=0, id_item=0):
    try:
        # 获取主页信息
        res_main = do_req(mian_url)
        # 获得页面评论文字
        soup = BeautifulSoup(res_main, "html.parser")
        # 获取评论条数
        totalnum = soup.select("div.mod-hd h2 span a")[0].get_text()[3:-2]
        print('共' + soup.select("div.mod-hd h2 span a")[0].get_text() + '项')
        # 计算出页数和最后一页的评论数
        pagenum = int(totalnum) // 20
        commentnum = int(totalnum) % 20
        print('共' + str(pagenum) + '页, 余' + str(commentnum) + '项')
    except:
        print("Requesting the main page failed!")
        checkpoint['id_page']=id_page
        checkpoint['id_item']=id_item
        return -1

    # 设置等待时间，避免爬取太快
    timeout = 3
    # 用于在超时的时候抛出异常，便于捕获重连
    socket.setdefaulttimeout(timeout)

    # 追加写文件的方式打开文件
    with open('./comment_full.csv', 'a+', encoding='utf-8') as cfile_full, open('./comment.csv', 'a+', encoding='utf-8') as cfile:
        writer_csv_full = csv.writer(cfile_full)
        # 循环爬取内容, 豆瓣只开放500条评论
        for page_id in range(id_page, min(10,pagenum)):
            print('第' + str(page_id) + '页')
            start = page_id * 20
            comment_url = MakeUrl(mian_url,start)

            # 超时重连
            state = False
            time_connect = 10
            while not state:
                try:
                    res_comment = do_req(comment_url)
                    state = True
                except socket.timeout:
                    if time_connect > 0:
                        state = False
                        time_connect -= 1
                    else:
                        print("Connection timeout!")
                        checkpoint['id_page']=page_id
                        checkpoint['id_item']=id_item
                        return -1
                except(error.HTTPError, error.URLError):
                    print("Connection failed!")
                    checkpoint['id_page']=page_id
                    checkpoint['id_item']=id_item
                    return -1

            # 获得评论
            soup = BeautifulSoup(res_comment, "html.parser")
            comments = soup.select("div.comment-item")
            # 判断页面为空
            if len(comments) == 0 or len(comments[0].select("div.avatar > a")) == 0:
                print("Comments empty!")
                checkpoint['id_page']=page_id
                checkpoint['id_item']=id_item
                return -1
            for item_id in range(id_item, len(comments)):
                item = comments[item_id]
                username = ''
                score = ''
                comment = ''
                pub_time = ''
                votes = ''
                # 获取评价人姓名
                item_username = item.select("div.avatar > a")
                if len(item_username) > 0:
                    username = item_username[0]['title']
                
                # 获取点赞数
                item_vote = item.select('div.comment > h3 > span.comment-vote > span[class="votes"]')
                if len(item_vote) > 0:
                    votes = item_vote[0].get_text()
                
                # 获取评价值
                # {5:"力荐",4:"推荐",3:"还行",2:"较差",1:"很差"}
                temp_score = item.select("div.comment > h3 > span.comment-info > span")
                str_score = ''
                for ts in temp_score:
                    str_score += str(ts)
                pattern = re.compile(r'<span class="allstar(.*?) rating" title="(.*?)"></span>')
                patter_score = pattern.findall(str_score)
                if patter_score != []:
                    score = str(int(patter_score[0][0])//10)

                # 获取评价内容
                temp_comment = item.select('div.comment > p > span[class="short"]')
                if len(temp_comment) > 0:
                    comment = temp_comment[0].get_text()
                    # 去掉多余空白
                    comment = re.sub(r'\s+',' ', comment)
                    """
                    # 只保留中文
                    comment = re.sub('[^\u4e00-\u9fa5]', '', comment)
                    # 分词，去掉停用词
                    temp_words = jieba.lcut(comment)
                    comment = ' '.join(list(set(temp_words)-set(stopwords)))
                    """
                
                # 获取评价时间
                temp_time = item.select('div.comment > h3 > span.comment-info > span[class="comment-time"]')
                if len(temp_time) > 0:
                    pub_time = temp_time[0]['title']
                
                # 写入文件
                writer_csv_full.writerow([username,score,comment,pub_time,votes])
                # 过滤空评论
                if comment!='':
                    if score != '':
                        cfile.write(score + '\t' + comment + '\n')
                    else:
                        with open('./comment_no_score.csv', 'a+', encoding='utf-8') as comment_no_score:
                            comment_no_score.write(comment + '\n')
            # 模拟人工
            time.sleep(random.uniform(0,3))

def get_movies_main_url(movies_url):
    # 登录
    # login()
    if os.path.exists('./checkpoint'):
        with open('./checkpoint', 'r', encoding='utf-8') as cp:
            checkpoint=json.loads(cp.read(),encoding='utf-8')
    print('Last point >>>',checkpoint)
    
    flag_begin = False
    flag_error = False
    res_list = do_req(movies_url)
    json_dict = json.loads(res_list)
    subjects = json_dict['subjects']
    for id_moive in range(checkpoint['id_moive'], len(subjects)):
        print([id_moive,subjects[id_moive]['title'],subjects[id_moive]['rate'],subjects[id_moive]['url']])
        if flag_begin:
            if get_movies_comments(subjects[id_moive]['url'], 0, 0) == -1:
                checkpoint['id_moive']=id_moive
                checkpoint['url_moive']=subjects[id_moive]['url']
                flag_error=True
                break
        else:
            flag_begin = True
            if get_movies_comments(subjects[id_moive]['url'], checkpoint['id_page'], checkpoint['id_item']) == -1:
                flag_error=True
                break
    with open('./checkpoint', 'w', encoding='utf-8') as cp:
        if not flag_error:
            checkpoint = {
                "id_moive":0,
                "id_page":0,
                "id_item":0,
                "url_moive":""
            }
        cp.write(json.dumps(checkpoint))

if __name__ == '__main__':
    # movies_url_hot = 'https://movie.douban.com/j/search_subjects?type=movie&tag=%E7%83%AD%E9%97%A8&sort=recommend&page_limit=500&page_start=0'
    # print('>>> 爬取豆瓣前500(实际324)部热门电影')
    # get_movies_main_url(movies_url_hot)

    # print('>>> 爬取豆瓣前500部最新电影')
    # movies_url_new = 'https://movie.douban.com/j/search_subjects?type=movie&tag=%E6%9C%80%E6%96%B0&sort=recommend&page_limit=500&page_start=0'
    # get_movies_main_url(movies_url_new)

    print('>>> 爬取豆瓣前500部评价最高电影')
    movies_url_best = 'https://movie.douban.com/j/search_subjects?type=movie&tag=%E8%B1%86%E7%93%A3%E9%AB%98%E5%88%86&sort=recommend&page_limit=500&page_start=0'
    get_movies_main_url(movies_url_best)
    print('Data crawling finished!')
