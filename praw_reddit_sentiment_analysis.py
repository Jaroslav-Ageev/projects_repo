import praw
import string
from nltk.corpus import stopwords
from nltk import FreqDist, bigrams

reddit = praw.Reddit(
    client_id="AV-p5TY0Ng6TFrwt6NKzxg",
    client_secret="jvn18NT4B9k6Tj5v7EZHYeiR2W6kiw",
    user_agent="python_book_demo"
)

post_text_list = []
comment_text_list = []
for post in reddit.subreddit("TellurianLNG").new(limit=20):
    post_text_list.append(post.selftext)
    # removes 'show more comments' instances
    post.comments.replace_more(limit=0)
    comments = post.comments.list()
    for c in comments:
        comment_text_list.append(c.body)

all_text = ' '.join(post_text_list + comment_text_list)

translator = str.maketrans('', '', string.punctuation + string.digits)
cleaned_text = all_text.translate(translator)

en_stopwords = set(stopwords.words('english'))
reddit_stopwords = set(['removed', 'dont']) | en_stopwords
cleaned_words = [w for w in cleaned_text.lower().split()
                 if w not in reddit_stopwords and len(w) > 3]

bg = [' '.join(bigr) for bigr in bigrams(cleaned_words)]
bg_fd = FreqDist(bg)
print(bg_fd.most_common(20))