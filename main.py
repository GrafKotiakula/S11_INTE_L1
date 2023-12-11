import os, sys
import telebot
import pandas as pd
import numpy as np
import random
from typing import Callable
from nbclassifier import NaiveBayesClassifier, __run as __runClassifier

TOKEN_DIR = 'TKN'
GAMES_DIR = 'GAMES'
DATASET_DIR = 'DATASET'
TRAINED_FILE = 'trained.txt'
YES = 0
NO = 1
MAYBE = 2

class GuessGameEngine:
    def __init__(self):
        self.__gameOver = True
    
    ''' Starts (restarts) game '''
    def startGame(self, gameData: pd.DataFrame, questionBuilder: Callable[[str], str]):
        self.__gameData = gameData.copy(deep=False)
        self.__questionBuilder = questionBuilder
        self.__curQuestion = None
        self.__gameSteps = 0
        self.__skippedQuestions = 0
        self.__gameOver = False
        self.__result = None
    
    def selectQuestion(self, reselect=False):
        if self.__gameOver:
            raise Exception('Game over')
        if reselect or self.__curQuestion == None:
            self.__gameSteps += 1
            entitiesLeft = self.__gameData.shape[0]
            questionLeft = self.__gameData.shape[1]
            if questionLeft > 0 and entitiesLeft > 1:
                sums = self.__gameData.sum().values
                chances = list(map(lambda x: 1 - abs(2 * x - entitiesLeft) / entitiesLeft, sums))
                sum_ch = sum(chances)
                if sum_ch > 0:
                    self.__curQuestion = random.choices(range(questionLeft), weights = chances, k=1)[0]
                else:
                    self.__curQuestion = random.choice(range(questionLeft))
            elif entitiesLeft > 0: # if no question left or only one entity left
                self.__curQuestion = random.choice(range(entitiesLeft))
                entity = self.__gameData.index[self.__curQuestion]
                return self.__questionBuilder(entity)
            else:
                self.__gameOver = True
                raise Exception('Entities ended')
        return self.__gameData.columns[self.__curQuestion]
    
    def answerQuestion(self, answer: int):
        if self.__curQuestion is None:
            raise Exception('No question to answer')
        if self.__gameData.shape[1] > 0 and self.__gameData.shape[0] > 1:
            question = self.__gameData.columns[self.__curQuestion]
            df = self.__gameData
            df = df.drop( df[ df[question] == answer ].index, axis=0 ) # drop rows
            self.__gameData = df.drop(question, axis=1) # drop question
        else:
            entity = self.__gameData.index[self.__curQuestion]
            if answer == YES:
                self.__gameOver = True
                self.__result = entity
            elif answer == NO:
                self.__gameData = self.__gameData.drop(entity, axis=0 ) # drop entity
        if answer == MAYBE:
            self.__skippedQuestions += 1
        if self.__gameData.shape[0] == 0:
            self.__gameOver = True
        self.__curQuestion = None
        return self.__gameOver
    
    def getGameSteps(self):
        return self.__gameSteps
    
    def getSkippedQuestions(self):
        return self.__skippedQuestions
    
    def isGameOver(self):
        return self.__gameOver
    
    def getResult(self, defaultAnswer:str = None):
        if self.__result is None:
            return defaultAnswer
        else:
            return self.__result        
    
    def overTheGame(self):
        self.__gameOver = True

def getTokenFromFile(name):
    with open(f'{TOKEN_DIR}/{name}.tkn', 'r') as f:
        return f.readline()

def __readDataset(srcFile:str):
    with open(f'{DATASET_DIR}/{srcFile}', 'r') as f:
        texts = []
        keys = []
        for l in f.readlines():
            text, key = l[:-1].split(':')
            texts.append(text)
            keys.append(key)
        return pd.DataFrame({'text': texts, 'class': keys})

def __readGameData(srcFile:str):
    with open(f'{GAMES_DIR}/{srcFile}', 'r') as f:
        entities, questions, relations = map( lambda txt: txt.split('\n'), f.read().split('\n\n') )
        relations = list(map( lambda l: list(map(lambda x: int(x), list(l))), relations ))
    result = pd.DataFrame(index=entities, columns=questions, data=relations)
    return result

def __splitList(l:list, n:int):
    k, m = divmod(len(l), n)
    return [l[i*k+min(i, m) : (i+1)*k+min(i+1, m)] for i in range(n)]

def __startGame(gge:GuessGameEngine, data:pd.DataFrame):
    gge.startGame(root_df, lambda e: f'Це {e}?')

def __summarizeGame(gge:GuessGameEngine):
    return f'''Поставлено запитань: {gge.getGameSteps()}
Пропущено запитань: {gge.getSkippedQuestions()}
*Відповідь: {gge.getResult("тварина мені не відома 😞")}*'''

def __transformInterpretation(inter:str):
    if inter == 'yes':
        return YES, 'Так'
    elif inter == 'no':
        return NO, 'Ні'
    else:
        return MAYBE, 'Мабуть'

if __name__ == '__main__':
    root_df = __readGameData('animals.txt')
    
    if '--train' in sys.argv or not os.path.exists(f'{DATASET_DIR}/{TRAINED_FILE}'):
        print('Reading training data...')
        dataset = __readDataset('ynm.data')
        nbc = NaiveBayesClassifier()
        
        print('Training...')
        nbc.learnFromDataFrame(dataset)
        nbc.saveToFile(f'{DATASET_DIR}/{TRAINED_FILE}')
    else:
        nbc = NaiveBayesClassifier(f'{DATASET_DIR}/{TRAINED_FILE}')
    
    if '--runc' in sys.argv:
        __runClassifier(nbc, stopword='стоп')
    
    if '--rung' in sys.argv:
        gge = GuessGameEngine()
        __startGame(gge, root_df)
        
        print('Select one from following animals:', root_df.index.to_list(), sep='\n')
        go = False
        while not gge.isGameOver():
            print(gge.selectQuestion())
            answ = input('> ')
            answ = nbc.analyze(answ)
            print(f'Your answer interpreted as {answ}')
            if answ == 'yes':
                answ = YES
            elif answ == 'no':
                answ = NO
            else:
                answ = MAYBE
            go = gge.answerQuestion(answ)
        
        print(f'Steps: {gge.getGameSteps()}, Skipped: {gge.getSkippedQuestions()}, Answer: {gge.getResult()}')
    
    if '--runb' in sys.argv or len(sys.argv) == 1:
        print('Creating telegram bot...')
        tgbot = telebot.TeleBot(getTokenFromFile('tgbot'))
        context = {}
        
        def getContext(chatId):
            if chatId not in context:
                context[chatId] = {'gge': GuessGameEngine()}
            return context[chatId]
        
        @tgbot.message_handler(commands=['start'])
        def start(msg: telebot.types.Message):
            tgbot.send_message(msg.chat.id, 'Привіт ✋. Це S11_INTLE_L1 бот. Напишіть /info для детльного опису')

        @tgbot.message_handler(commands=['info'])
        def info(msg: telebot.types.Message):
            text = '''Цей бот розроблений в рамках роботи над лабораторною роботою.
(Інформаційні технології)
Даний бот представляє гру, за правилами якої бот ви загадуєте якусь тварину із запропонованих, а бот намагається її відгадати.
Бот може ставити запитання на які вам потрібно відповідати "Так", "Ні", або "Не знаю".
Гра продовжуватиметься, доки бот не відгадає, або не здасться.
Для перегляду списку введіть команду /list
Для початку нової гри /start_game
Для дострокового завершення гри /end_game'''
            tgbot.send_message(msg.chat.id, text)

        @tgbot.message_handler(commands=['list'])
        def listEntities(msg: telebot.types.Message):
            colCount = 3
            entities = list(root_df.index)
            entities = __splitList(entities, colCount)
            lens = [len(max(c, key=len)) for c in entities]
            text = ''
            for i in range(len(entities[0])):
                line = ''
                for j in range(colCount):
                    if len(entities[j]) > i:
                        val = entities[j][i]
                    else:
                        val = ''
                    line += f' {val:<{lens[j]}} '
                line = line[:-1] + '\n'
                text += line
            text = f'Список тварин, які може вгадати бот:\n```List{text[:-1]}```'
            tgbot.send_message(msg.chat.id, text, parse_mode='Markdown')

        @tgbot.message_handler(commands=['start_game'])
        def startGame(msg: telebot.types.Message):
            gge = getContext(msg.chat.id)['gge']
            if gge.isGameOver():
                __startGame(gge, root_df)
                tgbot.send_message(msg.chat.id, f'Розпочнімо гру.\nМоє перше запитання:\n*{gge.selectQuestion()}*', parse_mode='Markdown')
            else:
                tgbot.send_message(msg.chat.id, 'Гра вже розпочата. Якщо хочете почати нову, спершу завершіть цю /end_game')
                tgbot.send_message(msg.chat.id, f'Моє поточне запитання:\n*{gge.selectQuestion()}*', parse_mode='Markdown')

        @tgbot.message_handler(commands=['end_game'])
        def endGame(msg: telebot.types.Message):
            gge = getContext(msg.chat.id)['gge']
            if gge.isGameOver():
                gge.startGame(root_df)
                tgbot.send_message(msg.chat.id, 'Гра не розпочата')
            else:
                tgbot.send_message(msg.chat.id, f'Гра завершена користувачем\n{__summarizeGame(gge)}', parse_mode='Markdown')
            tgbot.send_message(msg.chat.id, 'Для перегляду списку /list\nДля початку нової гри /start_game')

        @tgbot.message_handler(content_types=['text'])
        def text(msg: telebot.types.Message):
            gge = getContext(msg.chat.id)['gge']
            if gge.isGameOver():
                tgbot.send_message(msg.chat.id, 'Гра не розпочата\nДля перегляду списку /list\nДля початку нової гри /start_game')
            else:
                inter = nbc.analyze(msg.text)
                gameI, userI = __transformInterpretation(inter)
                gge.answerQuestion(gameI)
                if(gge.isGameOver()):
                    tgbot.reply_to(msg, f'_Вашу відповідь інтерпретовано як "{userI}"_\n\n{__summarizeGame(gge)}', parse_mode='Markdown')
                    tgbot.send_message(msg.chat.id, 'Для перегляду списку /list\nДля початку нової гри /start_game')
                else:
                    question = gge.selectQuestion()
                    tgbot.reply_to(msg, f'_Вашу відповідь інтерпретовано як "{userI}"_\n*{question}*', parse_mode='Markdown')
        
        print('Running telegram bot...')
        tgbot.infinity_polling()
