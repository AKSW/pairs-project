# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 12:28:07 2023

@author: monse
"""
import pandas as pd
from bs4 import BeautifulSoup
from dateparser import parse as parse_date
from urllib.request import urlopen
import requests

import logging

from pandas import DataFrame

logger = logging.getLogger("news")
logger.setLevel(logging.DEBUG)


count = 0

def get_content(link, para_length=3):
    total_paragraph_to_extract = para_length
    global count
    content = ''
    try:
        # data = urlopen(link)
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
        cookies = {'CONSENT': 'YES+cb.20220419-08-p0.cs+FX+111'}
        data = requests.get(link, headers=headers, cookies=cookies, timeout=10, verify=False)

        soup = BeautifulSoup(data.text, 'html.parser')
        pages = soup.find_all('p')
        for idx in range(0, min(total_paragraph_to_extract, len(pages))):
            content = content + ' ' + pages[idx].getText().strip()
    except:
        count += 1
    
    return content
#%%

def replace_empty_text(row):
    if row['text'] == '':
        return row['title']
    else:
        return row['text']

#%%

def get_articles_content(news_article_df, para_length=3):
    news_article_df['text'] = news_article_df['link'].apply(get_content, para_length=para_length)
    logger.debug('Total news articles: %d', len(news_article_df))
    logger.debug('Total articles with successful content: %d', (len(news_article_df) - count))
    logger.debug('total articles with failure count: %d', count)

    # when the content is not available, then put the title as content
    news_article_df['text'] = news_article_df.apply(replace_empty_text,axis=1)

    return news_article_df


import concurrent.futures
import urllib.request
def get_articles_content_parallel(news_article_df: DataFrame, para_length=3):

    # get URLs
    urls = news_article_df['link'].tolist()

    # Retrieve a single page and report the URL and contents
    def load_url(url, timeout):
        return get_content(url, 3)
        # with urllib.request.urlopen(url, timeout=timeout) as conn:
        #     return conn.read()

    texts = []
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Start the load operations and mark each future with its URL
        future_to_url = {executor.submit(load_url, url, 60): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (url, exc))
            else:
                print('%r page is %d bytes' % (url, len(data)))
                texts.append((url, data))

    text_df = pd.DataFrame(texts, columns=['url', 'text'])
    news_article_df = pd.merge(news_article_df, text_df, how='left', left_on="link", right_on="url")
    news_article_df = news_article_df[["title", "link", "published", "text"]]

    # when the content is not available, then put the title as content
    news_article_df['text'] = news_article_df.apply(replace_empty_text, axis=1)

    return news_article_df

if __name__ == "__main__":
    from crisisPrediction.google_news_util import decode_google_news_url
    news_article_df = pd.read_csv("1_google_news.csv")
    news_article_df['link'] = news_article_df['link'].apply(decode_google_news_url)
    news_article_df = get_articles_content_parallel(news_article_df)
    news_article_df.to_csv('2_google_news_paragraph.csv')
    
