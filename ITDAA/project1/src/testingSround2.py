import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 
import warnings 

from pathlib import Path
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist

warnings.filterwarnings('ignore')

#nltk.download('stopwords')
#nltk.download('wordnet')

from nltk.sentiment  import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


# 3.1. Preprocess your data as necessary and perform sentiment analysis on 
# the content column to categorize user reviews as positive, neutral, or negative. 
# Save the sentiments for each review in a new column called “Sentiment”

file_path= Path(r"../Project/data/chatgpt_reviews.csv")

df = pd.read_csv(file_path)
print(df['content'].head(50))

# Preprocess your data

def thumbup_count(text):
   return str(text).count("👍")

# count the number of thumbs up in the 'content' column and create a new column
df['thumbsUpCount'] = df['content'].apply(thumbup_count)

print(df['thumbsUpCount'].head())

    # preprocess CONTENT coloumn to only have the count ( bad , good , excl)

def rating_to_sentiment(stars):
    if stars >= 5:
        return "positive"
    elif stars == 4:
        return "positive"
    elif stars == 3:
        return "neutral"
    elif stars == 2:
        return "negative"
    elif stars == 1:
        return "negative"
    else:
        return "unknown"

sentiments_filtered = pd.DataFrame(df)
sentiments_filtered['sentiment'] = df['score'].apply(rating_to_sentiment)

df['sentiment'] = sentiments_filtered["sentiment"]
print(df.head(10))

    #Preprocess appVersion and reviewCreatedVersion null entries to previous coloumn's value

df.ffill(inplace=True)#forward fill

##
# 3.2.  Using visualizations, highlight features or issues that are frequently mentioned in
#  both positive and negative reviews. Additionally, identify which aspects users seem to 
#  love about ChatGPT vs which features they dislike.
##






























conn = sqlite3.connect('../Project/data/Q3_Output.db')      # change to csv
df.to_sql('clean_data', conn , if_exists='replace', index=False)
conn.close()

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(df.to_string())  













# Loading the dataset 
path = Path(r"../Project/data/Q3_Output.db")           # Change it to csv
#df = pd.read_csv(file_path)

conn = sqlite3.connect(path)
df = pd.read_sql_query("SELECT * FROM clean_data", conn)

print(df[df['sentiment'] == 'positive']['content'].head(10))

# Declare series variables for both positive and negative reviews
positive_reviews = df.loc[df['sentiment'] == 'positive', 'content']
print("Positive Reviews Before processing :")
print(positive_reviews)

negative_reviews = df.loc[df['sentiment'] == 'negative', 'content']



def process_sentences(text_series):
    """
    Cleans and preprocesses a Pandas Series of text data.
    Returns tokenized and lemmatized list of words for each sentence.
    """
    # 1. Remove special characters, numbers, and punctuation
    text_series = text_series.astype(str).replace(r"[^a-zA-Z#]", " ", regex=True)
    
    # 2. Remove short words (<= 3 characters)
    negation_words = {"not", "no", "never", "don’t", "can't", "isn't", "wasn't", "won't", "doesn't"}
    text_series = text_series.apply(
         lambda x: " ".join([
              w for w in x.split() if  w in negation_words or len(w) > 3]
              ))
    
    # 3. Tokenize the sentences
    tokenized = text_series.apply(lambda x: x.split())

    # 4. Remove stop words
    stop_words = set(stopwords.words('english'))
    tokenized = tokenized.apply(lambda x: [w for w in x if w.lower() not in stop_words])
    
    # 5. Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokenized = tokenized.apply(lambda x: [lemmatizer.lemmatize(w) for w in x])

    return tokenized


#**
#*      Exploartory Data analysis 

processed_positive = process_sentences(positive_reviews)
positive_words = " ".join([" ".join(sentence) for sentence in processed_positive]) 


print("Positive Reviews After processing :")
print(positive_words)

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(positive_words)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.title("Word Cloud: Positive Reviews", fontsize=16)
plt.show()



processed_negative = process_sentences(negative_reviews)
negative_words = " ".join([" ".join(sentence) for sentence in processed_negative]) 

wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(negative_words)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation= 'bilinear')
plt.axis('off')
plt.title("Word Cloud: Negative Reviews", fontsize=16)
plt.show()


#features or issues that are frequently mentioned in both positive and negative reviews.

            #positive results 

positive_words = [word for sentence in processed_positive for word in sentence]

fdist_positive = FreqDist(positive_words)
top_positive = FreqDist(positive_words).most_common(10)

print("Most Common Positive Words:")
for word, count in top_positive:
    print(f"{word}: {count}")

plt.figure(figsize=(8, 5))
sns.barplot(x=positive_words, y=top_positive, color='green')
plt.title("Top 10 Words in Positive Reviews")
plt.xlabel("Frequency")
plt.show()

            #negative results

negative_words = [word for sentence in processed_negative for word in sentence]

fdist_negative = FreqDist(negative_words)
top_negative = fdist_negative.most_common(10)

print("Most Common Negative Words:")
for word, count in top_negative:
    print(f"{word}: {count}")

plt.figure(figsize=(8, 5))
sns.barplot(x=negative_words, y=top_negative, color='red')
plt.title("Top 10 Words in Negative Reviews")
plt.xlabel("Frequency")
plt.show()


###
# which aspects users seem to  love about ChatGPT vs which features they dislike.
##

# Convert top word tuples into a DataFrame
pos_df = pd.DataFrame(top_positive, columns=['word', 'positive_count'])
neg_df = pd.DataFrame(top_negative, columns=['word', 'negative_count'])

# Merge on common words (to see overlap) or outer join for all
merged_df = pd.merge(pos_df, neg_df, on='word', how='outer').fillna(0)

# Sort by total mentions
merged_df['total'] = merged_df['positive_count'] + merged_df['negative_count']
merged_df = merged_df.sort_values('total', ascending=False)

# Plot
plt.figure(figsize=(12, 6))
merged_df.set_index('word')[['positive_count', 'negative_count']].plot(kind='bar', figsize=(12, 6), color=['green', 'red'])
plt.title('Top Words in Positive vs Negative Reviews')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y')
plt.show()






