# import library: bs4, urllib.request, numpy, string, stopwords
import bs4 as bs
import urllib.request
import numpy as np
import string
from nltk.corpus import stopwords
import nltk
from numpy import ndarray

# main class where we can train and predict data


class Main(object):
    # set parameters
  def __init__(self, epochs: int = 100):
    # V=Number of unique words in our corpus of text
    # W=Weights between input layer and hidden layer
    # W1=Weights between hidden layer and output layer
    # N=Number of neurons in the hidden layer of neural network
    self.epochs: int = epochs
    self.N: int = 10
    self.x_train: list[list[int]] = []
    self.y_train: list[list[int]] = []
    self.window_size: int = 2
    self.alpha: float = 0.001
    self.words: list[str] = []
    self.word_index: dict[str, int] = {}

  def initialize(self, data_len: int, data: list[str], load: bool = False):
    self.data_len = data_len
    if load:
      self.W = np.load('W.npy', allow_pickle=True)
      self.W1 = np.load('W1.npy', allow_pickle=True)
    else:
      self.W = np.random.uniform(-0.8, 0.8, (self.data_len, self.N))
      self.W1 = np.random.uniform(-0.8, 0.8, (self.N, self.data_len))

    self.words = data
    for i in range(len(data)):
      self.word_index[data[i]] = i

  # we calculate probability of word
  def probabilityOfWord(self, x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

  # we calculate Forward Propagation
  def propagate_forward(self, X: list[int]) -> ndarray:
    # <class 'numpy.ndarray'> (10, 1)
    self.h: ndarray = np.dot(self.W.T, X).reshape(self.N, 1)
    # <class 'numpy.ndarray'> (2957, 1)
    self.u: ndarray = np.dot(self.W1.T, self.h)
    # <class 'numpy.ndarray'> (2957, 1)
    self.y: ndarray = self.probabilityOfWord(self.u)
    return self.y

  # we calculate Back Propagation
  def propagate_backward(self, x: list[int], t: list[int]):
    e = self.y - np.asarray(t).reshape(self.data_len, 1)
    # e.shape is V x 1
    dLdW1: ndarray = np.dot(self.h, e.T)
    X = np.array(x).reshape(self.data_len, 1)
    dLdW = np.dot(X, np.dot(self.W1, e).T)
    self.W1: ndarray = self.W1 - self.alpha*dLdW1
    self.W: ndarray = self.W - self.alpha*dLdW

  # in this function we train our data
  # Kaisar
  def train(self):
    for epoch in range(1, self.epochs):
      self.loss = 0
      for j in range(len(self.x_train)):
        # Olzhas
        self.propagate_forward(self.x_train[j])
        # Olzhas
        self.propagate_backward(self.x_train[j], self.y_train[j])
        C = 0
        for m in range(self.data_len):
          if (self.y_train[j][m]):
            self.loss += -1*self.u[m][0]
            C += 1
        self.loss += C*np.log(np.sum(np.exp(self.u)))
      print(f"{epoch=}, {self.loss=}")
      self.alpha *= 1/((1+self.alpha*epoch))

  # in this function we predict words which similar and words often used together
  def predict(self, word: str, numberOfPredictions: int):
    if word in self.words:
      index = self.word_index[word]
      X = [0 for i in range(self.data_len)]
      X[index] = 1
      prediction = self.propagate_forward(X)
      output = {}
      for i in range(self.data_len):
        output[prediction[i][0]] = i

      topContextWords = []
      for k in sorted(output, reverse=True):
        topContextWords.append(self.words[output[k]])
        if (len(topContextWords) >= numberOfPredictions):
          break

      return topContextWords
    else:
      print("Word not found in dicitonary")

  def get_corpus(self) -> str:
    # with help of library urllib.request we can open website and do some operations like read
    data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')
    article = data.read()
    # with help of library bs4 we can take lxml format. lxml - the most feature-rich and easy-to-use library for processing XML and HTML in the Python language.
    parsedArticle = bs.BeautifulSoup(article, 'lxml')
    # from tag "p" we take all information
    paragraphs = parsedArticle.find_all('p')

    corpus = ""
    # we combine all text
    for p in paragraphs:
      corpus += p.text
    return corpus

  # we clean our data, we delete stopwords, space, punctuation, do lower words
  def pre_process(self, corpus: str) -> list[list[str]]:
    stop_words = set(stopwords.words('english'))
    data: list[list[str]] = []
    sentences = corpus.split(".")
    for i in range(len(sentences)):
      sentences[i] = sentences[i].strip()
      sentence = sentences[i].split()
      x = [word.strip(string.punctuation) for word in sentence
           if word not in stop_words]
      x = [word.lower() for word in x]
      data.append(x)
    return data

  def save(self) -> None:
    np.save('W.npy', self.W)
    np.save('W1.npy', self.W1)

  # this function help us to take our data and train it, we divide words from sentences, sort, count
  def train_test_split(self, sentences: list[list[str]], load: bool = False):
    data = {}
    for sentence in sentences:
      for word in sentence:
        if word not in data:
          data[word] = 1
        else:
          data[word] += 1
    data_len: int = len(data)
    data2: list[str] = sorted(list(data.keys()))
    vocab = {}
    for i in range(len(data2)):
      vocab[data2[i]] = i

    for sentence in sentences:
      for i in range(len(sentence)):
        center_word = [0 for x in range(data_len)]
        center_word[vocab[sentence[i]]] = 1
        context = [0 for x in range(data_len)]

        for j in range(i-self.window_size, i+self.window_size):
          if i != j and j >= 0 and j < len(sentence):
            context[vocab[sentence[j]]] += 1
        self.x_train.append(center_word)
        self.y_train.append(context)
    self.initialize(data_len, data2, load)

    return self.x_train, self.y_train


if __name__ == '__main__':
  try:
    nltk.data.find('stopwords')
  except Exception as e:
    nltk.download('stopwords')

  main = Main(epochs=100)
  # Arailym
  corpus = main.get_corpus()
  # Shynar
  data = main.pre_process(corpus)
  # Arailym
  main.train_test_split(data, True)
  # main.train_test_split(data)
  # Kaisar, Olzhas
  main.train()
  main.save()
  print(main.predict("artificial", 3))
