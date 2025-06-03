#
##          Question 3
##  Sentiment Analysis: ChatGPT user reviews
#
import pandas as pd 
import sqlite3
import seaborn as sns 
import string
import matplotlib.pyplot as plt 
import nltk

from textblob import TextBlob
from pathlib import Path
from nltk.sentiment  import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm 

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
sia = SentimentIntensityAnalyzer()



nltk.download('stopwords')
nltk.download('wordnet')


# 3.1. Preprocess your data as necessary and perform sentiment analysis on 
# the content column to categorize user reviews as positive, neutral, or negative. 
# Save the sentiments for each review in a new column called “Sentiment”

file_path= Path(r"../Project/data/chatgpt_reviews.csv")

df = pd.read_csv(file_path)
print(df['content'].head(5))



##             Preprocess data
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def process_sentences(text_series):
    """
    Cleans and preprocesses a Pandas Series of text data.
    Returns tokenized and lemmatized list of words for each sentence.
    """
    text_series = text_series.astype(str).replace(r"[^a-zA-Z#]", " ", regex=True)
    text_series = text_series.apply(lambda x: " ".join(w for w in x.split() if len(w) > 3))
    tokenized = text_series.apply(lambda x: [lemmatizer.lemmatize(w.lower()) for w in x.split() if w.lower() not in stop_words])
    return tokenized


##             sentiment score

def get_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


# Ensure content is string and handle NaNs
df['content'] = df['content'].astype(str).fillna("")

sentiments_filtered = pd.DataFrame(df)
sentiments_filtered['sentiment'] = df['content'].apply(get_sentiment)

df['sentiment'] = sentiments_filtered["sentiment"]
print(df.head(10))

def thumbup_count(text):
   return str(text).count("👍")
df['thumbsUpCount'] = df['content'].apply(thumbup_count)
print(df['thumbsUpCount'].head())

df.ffill(inplace=True)#forward fill :appVersion and reviewCreatedVersion null entries to previous coloumn's



###
##      3.2: Visualize liked vs Disliked Features
##       highlight features or issues that are frequently mentioned in both positive and negative reviews.
##       Additionally, identify which aspects users seem to love about ChatGPT vs which features they dislike.
###


#           Spilt Reviews 

positive_reviews = df[df['sentiment'] == 'positive']['content']
negative_reviews = df[df['sentiment'] == 'negative']['content']

processed_pos = process_sentences(positive_reviews)
processed_neg = process_sentences(negative_reviews)


#           Filter text

from nltk import FreqDist

positive_words = [word for sentence in processed_pos for word in sentence]
negative_words = [word for sentence in processed_neg for word in sentence]

positive_text = " ".join(positive_words)
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(positive_text)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.title("Word Cloud: Positive Reviews", fontsize=16)
plt.show()


negative_text = " ".join(negative_words)
wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(negative_text)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.title("Word Cloud: Negative Reviews", fontsize=16)
plt.show()


fdist_pos = FreqDist(positive_words)
fdist_neg = FreqDist(negative_words)

top_pos = fdist_pos.most_common(10)
top_neg = fdist_neg.most_common(10)



#
#       Create dataframe
#

df_pos = pd.DataFrame(top_pos, columns=["word", "count_pos"])
df_neg = pd.DataFrame(top_neg, columns=["word", "count_neg"])

        # Merge for comparison
merged_df = pd.merge(df_pos, df_neg, on="word", how="outer").fillna(0)
merged_df = merged_df.sort_values(by=["count_pos", "count_neg"], ascending=False)


        # Plot   Bar Graph Comparison 
plt.figure(figsize=(12,6))
bar_width = 0.4
x = range(len(merged_df))

plt.bar(x, merged_df["count_pos"], width=bar_width, label="Positive", color="green")
plt.bar([i + bar_width for i in x], merged_df["count_neg"], width=bar_width, label="Negative", color="red")

plt.xticks([i + bar_width / 2 for i in x], merged_df["word"], rotation=45)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Features Mentioned in Positive vs Negative Reviews")
plt.legend()
plt.tight_layout()
plt.show()



###
##      3.3. Critically discuss how user satisfaction scores have changed over time.
##       Motivate your answer using a line plot.
###


df['date'] = pd.to_datetime(df['date'])

df_monthly = df.groupby(df['date'].dt.to_period('M'))['score'].mean['score'].mean().reset_index()
df_monthly['date'] = df_monthly['date'].dt.to_timestamp()


plt.figure(figsize=(12,6))
sns.lineplot(data=df_monthly, x='date', y='score', marker='o')
plt.title("Average User Satisfaction Score Over Time")
plt.xlabel("Date")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



