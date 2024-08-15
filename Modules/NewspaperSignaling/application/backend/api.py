import datetime
from fastapi import FastAPI, status, Response
from fastapi.responses import JSONResponse
from pandas import DataFrame, Series
from pydantic import BaseModel, ConfigDict

import logging
import time

import json

from transformers import pipeline
import torch
from crisisPrediction.googlenewsapi import get_news
from crisisPrediction.news_articles_paragraph import get_articles_content, get_articles_content_parallel
import pandas as pd
from crisisPrediction.google_news_util import decode_google_news_url

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger("news")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

log_format = "%(asctime)s %(levelname)s -- %(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()


class EmptyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NewsArticleClassification(BaseModel):
    link: str
    text: str
    score: float


class ClassificationOutput(BaseModel):
    total_news: int
    num_high_alert: int
    num_low_alert: int
    num_others: int
    top_n_high_alert_results: list[NewsArticleClassification]



@app.get("/classify")
@app.post("/classify")
async def classify(query: str,
                   start_date: datetime.date,
                   end_date: datetime.date,
                   country="Germany",
                   language="de",
                   top_n: int = 5) -> ClassificationOutput:

    # Step 1: Get News Articles
    logger.debug("news retrieval")
    start = time.time()
    news_articles_df = analyze_news_articles(query, start_date, end_date, country, language)
    # news_articles_df = news_articles_df.head()
    end = time.time()
    logger.debug(f"execution time: {end - start}")

    # early termination if there were no news articles found
    if news_articles_df.empty:
        return Response(# content={"message": "No news articles found in the given time range."},
                            status_code=status.HTTP_204_NO_CONTENT)


    # Step 2: Chat Completion
    # generated_text = chat_completion(query, news_articles_df, openai_api_key)

    # Step 3: Similarity Check
    # cosined_articles_df = similarity_check(generated_text, news_articles_df, language)

    # Step 4: Tense Classification
    logger.debug("tense_classification")
    start = time.time()
    future_articles_df = classify_tense(news_articles_df)
    end = time.time()
    logger.debug(f"execution time: {end - start}")

    # Step 5: Alert Classification
    logger.debug("alert_classification")
    start = time.time()
    alert_dict = classify_alerts(future_articles_df)
    end = time.time()
    logger.debug(f"execution time: {end - start}")

    result = alert_dict
    result['top_n_high_alert_results'] = [NewsArticleClassification(link=r['link'],
                                                                    text=r['text'],
                                                                    score=r['alert_score'])
                                          for r in result['top_n_high_alert_results'][:top_n]]

    return ClassificationOutput(**result)


with open("config.json") as json_file:
    config = json.load(json_file)

# setup classification pipeline
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 10
classifier = pipeline("zero-shot-classification",
                      model=config['BERT_MODEL'],
                      batch_size=batch_size,
                      device=device
                      )


def analyze_news_articles(query: str,
                          start_date: datetime.date,
                          end_date: datetime.date,
                          country: str,
                          language: str,
                          use_titles_only: bool = False,
                          max_num_paragraphs: int = 3):
    # get news from RSS feed
    news = get_news(query, start_date, end_date, country, language)
    news_df = pd.DataFrame(news)
    # news_df = news_df.head(5)

    if news_df.empty:
        return news_df

    # resolve RSS link URLs
    if not use_titles_only:
        news_df['link'] = news_df['link'].apply(decode_google_news_url)

    # get article content
    news_article_df = get_articles_content_parallel(news_df, para_length=max_num_paragraphs)

    return news_article_df


def classify_tense(news_article_df: DataFrame, threshold: float = 0.7):
    categories_to_classify = ['past', 'present', 'future']
    text_column = "text"

    # classify the tense
    classification = classifier(news_article_df[text_column].tolist(), categories_to_classify, multi_label=False)

    # drop articles most likely referring to the past
    for idx, d in enumerate(classification):
        labels = d['labels']
        scores = d['scores']

        if (scores[labels.index('present')] + scores[labels.index('future')]) < threshold:
            news_article_df.drop(idx, inplace=True)

    return news_article_df


def classify_alerts(news_article_df: DataFrame) -> dict:
    categories_to_classify = ['Risk and Warning', 'Caution and Advice', 'Safe and Harmless']
    threshold = 0.50
    text_column = "text"

    # classify the alert level
    classification = classifier(news_article_df[text_column].tolist(), categories_to_classify, multi_label=False)

    # keep alert level with the highest confidence value
    for idx, d in enumerate(classification):
        id_max = Series(d['scores']).idxmax()

        news_article_df.at[idx, 'alert_class'] = d['labels'][id_max]
        news_article_df.at[idx, 'alert_score'] = d['scores'][id_max]

    alert_dict = dict()

    # get total length of news_df
    total_news = len(news_article_df)
    logger.debug('Total news: {}'.format(total_news))

    # total articles with alert_class = high alert
    num_high_alert = len(news_article_df[news_article_df['alert_class'] == 'Risk and Warning'])
    logger.debug('Risk and Warning: {}'.format(num_high_alert))

    # total articles with alert_class = low alert
    num_low_alert = len(news_article_df[news_article_df['alert_class'] == 'Caution and Advice'])
    logger.debug('Caution and Advice: {}'.format(num_low_alert))

    # total articles with alert_class = others
    num_others = len(news_article_df[news_article_df['alert_class'] == 'Safe and Harmless'])
    logger.debug('Safe and Harmless: {}'.format(num_others))

    # return the column text and alert_score for high alert articles with max and second max alert_score
    high_alert_df = news_article_df[news_article_df['alert_class'] == 'Risk and Warning']
    high_alert_df = high_alert_df.sort_values(by=['alert_score'], ascending=False)
    high_alert_df = high_alert_df[['link', 'text', 'alert_score']]
    high_alert_df = high_alert_df.reset_index(drop=True)

    # add to dictionary
    alert_dict['total_news'] = total_news
    alert_dict['num_high_alert'] = num_high_alert
    alert_dict['num_low_alert'] = num_low_alert
    alert_dict['num_others'] = num_others

    alert_dict['top_n_high_alert_results'] = high_alert_df.to_dict('records')

    return alert_dict


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    # @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8088)
