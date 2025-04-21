import streamlit as st
import pandas as pd
from GoogleNews import GoogleNews
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

class TextProcessing:
    def Newsfetch(self, keyword, period):
        gn = GoogleNews(period=period)
        gn.get_news(keyword)
        news = gn.results()

        df = pd.DataFrame(news)

        if df.empty:
            st.error("No data fetched. Please try again with different keywords or period.")
            return None
        
        st.write("Fetched Data Columns:", df.columns.tolist())  
        

        if 'title' in df.columns:
            st.success("Found 'title' column in the fetched data.")
        elif 'desc' in df.columns:
            st.warning("'title' column not found. Using 'desc' column instead.")
            df['title'] = df['desc'] 
        else:
            st.error("No 'title' or 'desc' column found in the fetched news data.")
            return None
        
        df.to_csv('news.csv', index=False)
        return df
    
    def SeparationandFetching(self):
        df = pd.read_csv("news.csv")
        if 'title' not in df.columns:
            raise KeyError("The 'title' column is not found in the CSV.")
        title = df["title"].str.lower()
        return title

    def textcleaningforvader(self, text):
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"[^\w\s.,!?;:\-()]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def Processedtitle(self, titles):
        filtercollector = []
        for title in titles:
            cleaned_title = self.textcleaningforvader(title)
            filtercollector.append(cleaned_title)
        return filtercollector

class Analyzing:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def SentimentScore(self, titles):
        data_positive = []
        data_negative = []
        data_neutral = []
        for title in titles:
            scores = self.sia.polarity_scores(title)
            compound = scores['compound']
            title_data = {
                'title': title,
                'neutral': scores['neu'],
                'positive': scores['pos'],
                'negative': scores['neg'],
                'compound': compound
            }
            

            if compound > 0.05:
                data_positive.append(title_data)
            elif compound < -0.05:
                data_negative.append(title_data)
            else:
                data_neutral.append(title_data)


        df_positive = pd.DataFrame(data_positive)
        df_negative = pd.DataFrame(data_negative)
        df_neutral = pd.DataFrame(data_neutral)

        
        st.write("Positive Titles:")
        st.dataframe(df_positive)  
        st.write("Negative Titles:")
        st.dataframe(df_negative)  
        st.write("Neutral Titles:")
        st.dataframe(df_neutral)  

        
        df_positive.to_csv('positive_titles.csv', index=False)
        df_negative.to_csv('negative_titles.csv', index=False)
        df_neutral.to_csv('neutral_titles.csv', index=False)

        return df_positive, df_negative, df_neutral


class Visualization:
    def SentimentPieChart(self, df_positive, df_negative, df_neutral):
        categories = {
            'positive': len(df_positive),
            'negative': len(df_negative),
            'neutral': len(df_neutral),
        }


        st.write("Sentiment Categories Count:", categories)


        counts = list(categories.values())
        labels = list(categories.keys())
        fig, ax = plt.subplots()
        ax.pie(counts, labels=labels, autopct='%1.1f%%')
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)

def main():
    st.title("Sentiment Analysis of News Titles")
    
    textprocessing = TextProcessing()
    analyzing = Analyzing()
    visualization = Visualization()

    
    keyword = st.text_input("Enter the keyword to search for news:")
    period = st.text_input("Enter the period for news (e.g., '7d' for 7 days):")

    if st.button("Fetch and Analyze News"):
        if keyword and period:  
            df = textprocessing.Newsfetch(keyword, period)
            if df is None:
                return 

            titles = textprocessing.SeparationandFetching()
            cleaned_titles = textprocessing.Processedtitle(titles)

            df_positive, df_negative, df_neutral = analyzing.SentimentScore(cleaned_titles)
            st.success("Sentiment analysis completed.")

            visualization.SentimentPieChart(df_positive, df_negative, df_neutral)
        else:
            st.error("Please enter both keyword and period.")

if __name__ == "__main__":
    main()