import flair

def funnyScorer(text: str) -> bool:
  flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
  s = flair.data.Sentence(text)
  flair_sentiment.predict(s)
  total_sentiment = s.labels
  print(total_sentiment)
  return 

funnyScorer("It is 12:00AM on November 1, 2023")