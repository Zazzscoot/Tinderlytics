import flair
from matplotlib import pyplot as plt
from jokes import funny_jokes, sarcastic_jokes, diverse_jokes


def funnyScorer(text: str) -> bool:
  flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
  s = flair.data.Sentence(text)
  flair_sentiment.predict(s)
  total_sentiment = s.labels
  neg = str(total_sentiment[0]).split('(')[0][-9:-1] == 'NEGATIVE'
  if neg:
     return -1 * float(str(total_sentiment[0]).split("(")[1].split(")")[0])
  else: 
     return float(str(total_sentiment[0]).split("(")[1].split(")")[0])

def test_jokes(jokes: list) -> None:
  allFunny = []
  for joke in jokes:
      allFunny.append(funnyScorer(joke))
 
  print(f"The average sentiment score is: {sum(allFunny) / len(allFunny)}")
  plt.hist(allFunny)
  plt.show()

if __name__ == "__main__":
  #print(funnyScorer("The Tinder bio 'Looking for a fwb' is not suitable for a long-term relationship. It is important to find a partner who is compatible with you and has similar values. A fling is not what you are looking for, and you should be honest with potential partners about what you are looking for. Is there anything else I can help you with?"))
  test_jokes(sarcastic_jokes)