import nltk
import random
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize lemmatizer
lemmer = WordNetLemmatizer()

# Sample knowledge base
corpus = """
Hello! I am an AI chatbot.
I can answer your basic questions.
Hi there! How can I help you today?
Greetings! Feel free to ask me anything.

Python is a popular programming language.
It is used for web development, data science, and AI.
Java is also a widely used programming language.
C is a foundational programming language.

Machine learning is a part of artificial intelligence.
Artificial intelligence enables machines to think and learn.
Deep learning is a subset of machine learning.
Data science involves analyzing data to gain insights.

NLP stands for Natural Language Processing.
NLP helps computers understand human language.
Chatbots use NLP to interact with users.

An internship helps students gain real-world experience.
Internships improve practical skills.
Students can learn industry knowledge through internships.

HTML is used to create web pages.
CSS is used for styling web pages.
JavaScript adds interactivity to websites.

A database stores organized information.
SQL is used to manage databases.
MySQL is a popular database system.

Cloud computing provides online storage and services.
AWS is a cloud platform.
Azure is also a cloud service provider.

Cybersecurity protects systems and data from attacks.
Encryption is used to secure data.
Passwords should be strong and unique.

Software development involves designing and building applications.
Testing ensures software quality.
Debugging fixes errors in programs.

Operating systems manage computer hardware.
Windows and Linux are popular operating systems.
Android is a mobile operating system.

Good communication skills are important for jobs.
Time management helps in productivity.
Teamwork is essential in workplaces.

Thank you for chatting with me.
Have a great day!
Goodbye and take care!
"""

# Preprocess text
sentence_tokens = nltk.sent_tokenize(corpus)

# Remove punctuation
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

# Lemmatization function
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# Normalize text
def LemNormalize(text):
    return LemTokens(word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting inputs and responses
GREETING_INPUTS = ("hello", "hi", "hey")
GREETING_RESPONSES = ["Hello!", "Hi there!", "Hey!", "Hi :)"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Response function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_input):
    sentence_tokens.append(user_input)

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize)
    tfidf = TfidfVec.fit_transform(sentence_tokens)

    similarity = cosine_similarity(tfidf[-1], tfidf)
    idx = similarity.argsort()[0][-2]
    flat = similarity.flatten()
    flat.sort()
    score = flat[-2]

    if score < 0.1:
        bot_response = "Sorry, I don't understand."
    else:
        bot_response = sentence_tokens[idx]

    sentence_tokens.pop()  # remove user input after processing

    return bot_response

# Chat loop
print("Chatbot: Hello! Type 'bye' to exit.")

while True:
    user_input = input("You: ").lower()

    if user_input == 'bye':
        print("Chatbot: Goodbye!")
        break

    elif greeting(user_input) is not None:
        print("Chatbot:", greeting(user_input))

    else:
        print("Chatbot:", response(user_input))