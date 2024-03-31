from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TranscriptSentimentizer:
    def __init__(self):
        self.model = SentimentIntensityAnalyzer()

    def apply(self, text: str) -> str:
        vs = self.model.polarity_scores(text)
        # consolidate to 1 sentiment, bagging scores between -0.5 and 0.5 as neutral
        # see: https://vadersentiment.readthedocs.io/en/latest/pages/about_the_scoring.html
        if vs['compound'] >= 0.5:
            sentiment = "positive"
        elif vs['compound'] <= -0.5:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        return sentiment
