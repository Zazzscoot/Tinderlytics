import flair
from matplotlib import pyplot as plt
from jokes import funny_jokes, sarcastic_jokes, diverse_jokes


def funnyScorer(text: str) -> bool:
  flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
  s = flair.data.Sentence(text)
  flair_sentiment.predict(s)
  total_sentiment = s.labels
  return float(str(total_sentiment[0]).split("(")[1].split(")")[0])

def test_jokes(jokes: list) -> None:
  allFunny = []
  for joke in jokes:
      allFunny.append(funnyScorer(joke))
 
  print(f"The average sentiment score is: {sum(allFunny) / len(allFunny)}")
  plt.hist(allFunny)
  plt.show()

if __name__ == "__main__":
  test_jokes(sarcastic_jokes)
  test_jokes(funny_jokes)
  test_jokes(diverse_jokes)