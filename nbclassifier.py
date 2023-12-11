import sys, re, os
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS

''' 
    Naive Bayes Classifier with natural language processing (NLP) 
'''

class NaiveBayesClassifier:
    def __init__(self, fileName:str = None):
        self.resetModel()
        if fileName:
            self.__readFromFile(fileName)
        self.__lemmatizer = WordNetLemmatizer()
    
    def resetModel(self):
        self.__totalEntries = 0
        self.__classes = {} # name - description (dict)
        self.__words = {} # name - description (dict)
    
    def __addClazz(self, clazz:str, wordsCount:int):
        if clazz not in self.__classes:
            self.__classes[clazz] = {'count': 1, 'words': wordsCount}
            self.__words = {k: {**v, clazz: 0} for k, v in self.__words.items()}
        else:
            self.__classes[clazz]['count'] += 1
            self.__classes[clazz]['words'] += wordsCount
        
    def __addWord(self, word:str, clazz:str):
        if word not in self.__words:
            self.__words[word] = dict( zip(self.__classes.keys(), [0] * len(self.__classes)) )
        self.__words[word][clazz] += 1
    
    def __extractPureWords(self, text:str):
        tokens = word_tokenize(text)
        words = [w for w in tokens if re.match('.*[\w].*', w)]
        return [self.__lemmatizer.lemmatize(w.lower()) for w in words]
    
    def add(self, clazz:str, description:str):
        words = self.__extractPureWords(description)
        self.__totalEntries += 1
        self.__addClazz(clazz, len(words))
        for w in words:
            self.__addWord(w, clazz)
    
    def analyzeGetAll(self, text:str):
        result = {}
        savedWordsCount = len(self.__words)
        words = self.__extractPureWords(text)
        
        for c, c_desc in self.__classes.items():
            tmp = c_desc["count"] / self.__totalEntries
            for w in words:
                if w in self.__words:
                    tmp = tmp * (self.__words[w][c] + 1) / (c_desc["words"] + savedWordsCount)
            result[c] = tmp
        
        return dict( sorted(result.items(),key=lambda item:item[1], reverse=True) )
    
    def analyze(self, text:str):
        return list(self.analyzeGetAll(text).keys())[0]
    
    def learnFromDataFrame(self, data:pd.DataFrame):
        for _, row in data.iterrows():
            self.add(row['class'], row['text'])
    
    # returns percent of success guessing ; dictionary {class: (successPercent, isKnown)}
    def testFromDataFrame(self, data:pd.DataFrame):
        classErr = {c: [0, 0, True] for c in self.__classes} # name: [err, total, isKnown]
        for _, row in df_test.iterrows():
            clazz = row['class']
            text = row['text']
            analyzedClazz = self.analyze(text)
            if clazz not in classErr:
                classErr[clazz] = [0, 0, False]
            classErr[clazz][1] += 1
            if not analyzedClazz == clazz:
                classErr[clazz][0] += 1
        
        resultErrPercent = (1 - sum([v[0] for v in classErr.values()]) / sum([v[1] for v in classErr.values()])) * 100
        resultClassErrPercent = {c: ((1 - v[0] / v[1]) * 100, v[2]) for c, v in classErr.items()}
        
        return resultErrPercent, dict(sorted(resultClassErrPercent.items(), key=lambda item: item[1]))
    
    def saveToFile(self, fileName:str):
        with open(fileName, 'w') as f:
            f.write(str(self.__totalEntries))
            f.write('\n')
            f.write( ' '.join(self.__classes.keys()) )
            f.write('\n')
            f.write(' '.join( [str(v["count"]) for v in self.__classes.values()] ))
            f.write('\n')
            f.write(' '.join( [str(v["words"]) for v in self.__classes.values()] ))
            f.write('\n')
            for w, w_desc in self.__words.items():
                f.write(w)
                f.write(' ')
                f.write(' '.join( [str(w_desc[c]) for c in self.__classes] ))
                f.write('\n')
    
    def __readFromFile(self, fileName:str):
        with open(fileName, 'r') as f:
            totalEntities, classNames, classCounts, classWordCounts, *words = f.readlines()
        self.__totalEntries = int(totalEntities)
        classNameList = classNames[:-1].split(' ')
        self.__classes = dict(zip( classNameList, [ {'count':int(c), 'words':int(w)} 
                                                   for c, w in zip(classCounts.split(' '), classWordCounts.split(' ')) ] ))
        for wl in words:
            word, *counts = wl.split(' ')
            self.__words[word] = dict(zip( classNameList, [int(c) for c in counts] ))
            
    def __str__(self):
        return f'totalEntries: {self.__totalEntries}\nclasses({len(self.__classes)}): {self.__classes}\nwords({len(self.__words)}): {self.__words}'
    
    def showWordCloud(self, width:int = 1000, height:int = 1000, stopwords:list[str]=STOPWORDS):
        wordsFrequencies = {k: sum(v.values()) for k, v in self.__words.items() if k not in stopwords}
        wordcloud = WordCloud(width=width, height=height, 
                              background_color='white').generate_from_frequencies(wordsFrequencies)

        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.gcf().canvas.manager.set_window_title('Word cloud')
        plt.show()
    
    def getClasses(self):
        return list(self.__classes.keys())
    
    def getTotalEntries(self):
        return self.__totalEntries
    
    def getWords(self):
        return {word : sum(classes.values()) for word, classes in self.__words.items()}

def __parseDataset(srcFile:str, testPartRatio:float):
    df_train = pd.read_csv(srcFile)
    df_train.rename(columns={'blurb': 'text', 'state': 'class'}, inplace=True)
    df_train.drop(columns = df_train.columns[0], axis=1, inplace=True)
    df_train.dropna(axis=0, inplace=True)
    
    df_test = df_train.sample(frac=testPartRatio)
    df_train.drop(df_test.index, inplace=True)
    
    return df_train, df_test

def __checkArgs():
    print()
    if len(sys.argv) == 1:
        print('program uses native bayes for classifying text')
        print('it studies on given dataset (default: https://www.kaggle.com/datasets/oscarvilla/kickstarter-nlp)')
        print('use arguments to specify task:\n')
        print('\t--describe to describe trained model')
        print('\t--test to test model')
        print('\t--word-cloud to show word cloud of trained model')
        print('\t--run to use model with custom input')
        print()
        print('\t--all to do all above')
        print()
        print('\t--train to force model train again')
        print()

def __describeClassifier(classifier: NaiveBayesClassifier, stopwords:list[str] = STOPWORDS, wordCount:int = 10):
    words = classifier.getWords()
    print(f'Total entries analyzed: {classifier.getTotalEntries()}')
    print(f'Classifier classes: {classifier.getClasses()}')
    print(f'Total learned words: {len(words)}')
    print(f'{wordCount} most common words:')
    print(f' {"Word":<10} | Frequency ')
    print(f'={"":=<10}=|===========')
    wordsToPrint = sorted([(w, c) for w, c in words.items() if w not in stopwords], key= lambda items: items[1], reverse=True)[:wordCount]
    for word, count in wordsToPrint:
        print(f' {word:>10} | {count}')
        
    print()
    
def __printPrecision(classifier: NaiveBayesClassifier, df_test:pd.DataFrame):
    print('Testing...')
    total, classes = classifier.testFromDataFrame(df_test)
    
    print(f'Total entries tested: {len(df_test.index)}')
    print(f'Total precision : {total:.3f}%')
    print(f' {"Class":<10} | Precision')
    print(f'============|===========')
    for c, v in classes.items():
        print(f'{c:>11} | {v[0]:>7.3f} %')
    print()

def __run(classifier: NaiveBayesClassifier, stopword = 'quit', inputFlag = '> '):
    print(f'Classifying user sentences. Input your sentence or \'{stopword}\' for quit.')
    sentence = input(inputFlag)
    while sentence != stopword:
        result = classifier.analyzeGetAll(sentence)
        for k, v in result.items():
            print(f'\t{k:>11} | {v}')
        sentence = input(inputFlag)
    print()
    
if __name__ == '__main__':
    dataset = 'dataset'
    output = 'output'
    
    if not os.path.isdir(dataset):
        os.makedirs(dataset)
    
    if not os.path.isdir(output):
        os.makedirs(output)
    
    if '--train' in sys.argv or not os.path.exists(f'{output}/test.csv') or not os.path.exists(f'{output}/trained.txt'):
        print()
        print('Reading data...')
        
        df_train, df_test = __parseDataset(f'{dataset}/df_text_eng.csv', 0.1)
        df_test.to_csv(f'{output}/test.csv')
        nbc = NaiveBayesClassifier()
        
        print('Training...')
        nbc.learnFromDataFrame(df_train)
        nbc.saveToFile(f'{output}/trained.txt')
    else:
        nbc = NaiveBayesClassifier(f'{output}/trained.txt')
        df_test = pd.read_csv(f'{output}/test.csv')
    
    __checkArgs()
    
    if '--describe' in sys.argv or '--all' in sys.argv:
        __describeClassifier(nbc)
    
    
    if '--test' in sys.argv or '--all' in sys.argv:
        __printPrecision(nbc, df_test)
    
    if '--word-cloud' in sys.argv or '--all' in sys.argv:
        nbc.showWordCloud()
    
    if '--run' in sys.argv or '--all' in sys.argv:
        __run(nbc)
   