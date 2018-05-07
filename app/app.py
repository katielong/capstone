from flask import Flask, render_template, request, redirect, flash
import pandas as pd
import requests
import re
import os
import dill
import networkx as nx
import operator
import time
import numpy as np
from datetime import datetime

from bs4 import BeautifulSoup

from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, DataRange1d, Plot, LinearAxis
from bokeh.embed import components
from bokeh.core.properties import value
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models.graphs import from_networkx
from bokeh.models.glyphs import HBar

from sklearn import base, model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

G = dill.load(open('static/G.pkd', 'r'))
centrality = dill.load(open('static/centrality.pkd', 'r'))
full_est = dill.load(open('static/full_est.pkd', 'r'))
userList = pd.read_csv('static/nodes.csv')
tagRepository = pd.read_csv('static/tag_category.csv', header=None)[1].tolist()
now = datetime.fromtimestamp(time.time()).strftime('%m/%d/%y %I:%M %p')
userIds = sorted(userList.id.tolist())
error = None
recodeDay = {
  'Monday':0, 
  'Tuesday':1,
  'Wednesday':2,
  'Thursday':3, 
  'Friday':4, 
  'Saturday':5, 
  'Sunday':6
}
toptags =  ['.net', 'ajax', 'algorithm',
            'asp.net', 'asp.net-mvc', 'c', 'c#', 'c++', 'css', 'database',
            'html', 'iphone', 'java', 'javascript', 'jquery', 'linq',
            'linux', 'multithreading', 'mysql', 'performance', 'php',
            'python', 'regex', 'ruby', 'ruby-on-rails', 'security', 'sql',
            'sql-server', 'svn', 'unit-testing', 'vb.net', 'visual-studio',
            'visual-studio-2008', 'web-services', 'windows', 'winforms', 'wpf',
            'xml']
Xcolumns = ['q_body_len', 'q_day', 'q_hour', '.net', 'ajax', 'algorithm',
            'asp.net', 'asp.net-mvc', 'c', 'c#', 'c++', 'css', 'database',
            'html', 'iphone', 'java', 'javascript', 'jquery', 'linq',
            'linux', 'multithreading', 'mysql', 'performance', 'php',
            'python', 'regex', 'ruby', 'ruby-on-rails', 'security', 'sql',
            'sql-server', 'svn', 'unit-testing', 'vb.net', 'visual-studio',
            'visual-studio-2008', 'web-services', 'windows', 'winforms', 'wpf',
            'xml', 
            'degree', 'in degree', 'out degree', 'degree centrality',
            'in degree centrality', 'out degree centrality', 'allN', 'outN',
            'inN', 'avg_n_contact', 'avg_n_in_contact', 'avg_n_out_contact']

# FUNCTIONS
def checkID(id):
  if id in userIds:
    return True
  else:
    return False

def wordBag(text):
  wordCount = dict()
  for word in re.findall("\w+", text):
    w = word.lower()
    if w not in wordCount:
      wordCount[w] = 1
    else:
      wordCount[w] += 1
  return sorted(wordCount.items(), key=operator.itemgetter(1), reverse=True)

def findTags(words, repository):
  tags = []
  for word, count in words:
    if word in repository:
      tags.append(word)
  return tags

def fillTags(target, source, currenttags):
  for i, d in enumerate(source):
    if d in currenttags:
      target[i] = 1
  return target

def extractTime(text):
  dt = datetime.strptime(text, '%m/%d/%y %I:%M %p')
  hour = datetime.strftime(dt, '%H')
  day = datetime.strftime(dt, '%A')
  return {'day': recodeDay[day], 'hour': int(hour)}

def drawNetwork(G, id):
  neighbors = [i for i in nx.all_neighbors(G, id)] 
  p = figure(title='Ego Network of User ' + id, x_range = (-1.1, 1.1), y_range = (-1.1, 1.1),
                tools='', toolbar_location=None, plot_height = 400,plot_width=500)   
  ego = nx.ego_graph(G, id)
  graph = from_networkx(ego, nx.spring_layout, scale=2, center = (0,0))
  p.renderers.append(graph)
  script, div = components(p)
  return {'script': script, 'div': div}

def drawWordFreq(words):
  plotData = {}
  words = words[0:10]
  word = [w for w, c in words]
  count = [c for w, c in words]
  plotData['word'] = word
  plotData['count'] = count
  source = ColumnDataSource(plotData)
  p = figure(x_range = word, title='Word Count', plot_height = 400, plot_width=500, toolbar_location=None, tools="")
  p.vbar(x='word', top='count', source = source, width = 0.3)
  script, div = components(p)
  return {'script': script, 'div': div}

# LOAD APP
@app.route('/', methods=['GET', 'POST'])
def index():
  error = None
  return render_template('index.html', now = now, error = error)

@app.route('/output', methods=['GET', 'POST'])
def output():
  # Get Input
  postText = request.form.get('postText')
  userId = request.form.get('userId')
  postTime = request.form.get('postTime')

  if checkID(userId) and postText != '' and postTime != '':
    features = []
    X = pd.DataFrame(columns = Xcolumns)
    networkG = drawNetwork(G, userId)
    html = ' '.join([p.text for p in BeautifulSoup(postText, 'lxml').find_all('p')])
    tag = ','.join(findTags(wordBag(html), tagRepository))

    # 'q_body_len', 'q_day', 'q_hour'
    q_body_chars = len(html)
    q_body_words = len(re.findall('\w+', html))
    q_hour = extractTime(postTime)['hour']
    q_day = extractTime(postTime)['day']
    features.extend([q_body_words, q_day, q_hour])
    # if in tags
    inToptags = fillTags([0]*len(toptags), toptags, tag)
    features.extend(inToptags)
    # network status
    network = centrality.loc[centrality.index == userId,]
    degree = int(network['degree'].values[0])
    cent = '{:.5f}'.format(float(network['degree centrality'].values[0]))
    features.extend(network.values.tolist()[0])
    #add all features to X
    X.loc[-1] = features
    # Make Prediction
    # fa: First Answer
    fa_s = full_est.predict(X)[0]
    fa_h = fa_s/3600
    fa_d = fa_h/24

    #draw bar chart
    bar = drawWordFreq(wordBag(html))
    return render_template('output.html', q_body_words = q_body_words, postText = postText, html = html, q_body_chars = q_body_chars, userId = userId, 
                            scriptBar = bar['script'], divBar = bar['div'], scriptNW = networkG['script'], divNW = networkG['div'], 
                            centrality=cent, degree = degree, tag = tag, fa_s = fa_s, fa_h = fa_h, fa_d = fa_d)
  else:
    error = 'All fields have to be filled'
    return render_template('index.html', now = now, error = error)

if __name__ == '__main__':
  app.run(port=8888)