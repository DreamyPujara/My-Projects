import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    articles = soup.find_all('article')
    news_data = []
    for article in articles:
        title = article.find('h2').get_text()
        content = article.find('p').get_text()
        news_data.append({'title': title, 'content': content})
    return pd.DataFrame(news_data)

# Example URL
url = 'https://www.example-news-website.com'
news_df = fetch_news(url)
news_df.to_csv('news_data.csv', index=False)
