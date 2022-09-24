import nltk
import pandas as pd

def process_data(df):   
    rows = []
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for i in range(len(df['summary'])):
        summary = df.loc[i]['summary']
        token = nltk.tokenize.word_tokenize(summary)
        stemmed = [stemmer.stem(word) for word in nltk.tokenize.word_tokenize(summary)]
        lemmatized = [lemmatizer.lemmatize(word) for word in nltk.tokenize.word_tokenize(summary)]
        row = [summary, token, stemmed, lemmatized]
        rows.append(row)
    return rows


if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    df = pd.read_csv('Musical_instruments_reviews.csv')
    processed_df = process_data(df)
    for row in processed_df:
        print("Original: ", row[0], "\n Tokenized: ", row[1], "\n Stemmed: ", row[2], "\n Lemmatized: ", row[3], "\n \n")