# This part is to fetch Feed data from Facebook Graph API and store it to feed_data.json and feed_data.csv

import json
import csv
from urllib.request import urlopen
import numpy as np
import pandas as pd
import math
from behavioural.sentimentalModel import sentimentalAnalysis as senti

def Senti_Extractor(num):
    if num>0:
        return math.ceil(num)
    else:
        return math.floor(num)

def feed_score(s):
    with urlopen(
     "https://graph.facebook.com/v6.0/me?fields=id%2Cname%2Cfeed%7Bcreated_time%2Cdescription%7D%2Cposts%7Bcreated_time%2Cmessage%2Cdescription%2Ccaption%7D&access_token={}".format(s)) as response:
        source = response.read()

    data = json.loads(source)

    some_data = data['feed']

    outfile = open("feed_data.csv", "w")

    fieldNames = ['created_time', 'id', 'description']

    writer = csv.DictWriter(outfile, fieldnames = fieldNames)
    writer.writeheader()

    while len(some_data['data']) != 0:
        with open('feed_data.json', 'a+', encoding='utf-8') as f:
            json.dump(some_data['data'], f, ensure_ascii=False, indent=4)

        for row in some_data['data']:
            writer.writerow(row)

        paging_next = some_data['paging']['next']

        with urlopen(paging_next) as response:
            source = response.read()
        some_data = json.loads(source)
        
    senti_value = pd.Series([])
    confidence = pd.Series([])
    
    df = pd.read_csv("feed_data.csv")
    df = df[pd.notnull(df['description'])]
    df['Response'] = np.nan
    df.reset_index(drop=True, inplace=True)
    df.to_csv("feed_clean.csv")
    
    for i in range(len(df)):
        temp = []
        column = df['description'][i]
        temp.append(column)
        senti_dict = senti(temp)
        senti_value[i] = Senti_Extractor(senti_dict.get('compound'))
        confidence[i] = senti_dict.get('compound')
        
    # Inserting Senti_Value and Confidence column

    df.insert(4, "Senti_Value", senti_value)
    df.insert(5, "Confidence", confidence)
    df.drop(columns = ['Response'], inplace = True)
    
    # Assigning weightage to DataTime

    total_rows = len(df.index)
    even = False

    time_weigh = pd.Series([])

    if total_rows % 2 == 0:
        even = True

    if even == True:
        i = total_rows/2 - 1
        time_weigh[i] = 25
        i = i-1
        while(i >= 0):
            time_weigh[i] = time_weigh[i+1]/2
            i = i-1

        i = total_rows/2
        time_weigh[i] = 25
        i = i+1
        while(i < total_rows):
            time_weigh[i] = time_weigh[i-1]/2
            i = i+1
    else:
        i = total_rows/2 - 0.5
        time_weigh[i] = 50/3
        time_weigh[i+1] = 50/3
        time_weigh[i-1] = 50/3
        time_weigh[i-2] = 25/2
        i = i - 3
        while (i >= 0):
            time_weigh[i] = time_weigh[i+1]/2
            i = i - 1

        i = total_rows/2 - 0.5
        time_weigh[i+2] = 25/2
        i = i+3
        while (i < total_rows):
            time_weigh[i] = time_weigh[i-1]/2
            i = i+1

    # Inserting Time_Weight

    df.insert(2, "Time_Weight", time_weigh)
    # Ranking based on Confidence

    df['Confi_Rank'] = df['Confidence'].rank(ascending = 0, method = 'dense')
    df = df.set_index('Confi_Rank')
    df = df.sort_index()

    # Assigning weightage based on Confidence
    confi_weigh = pd.Series([])

    confi_weigh[1] = 50

    for ind in df.index:
        if (ind != 1.0):
            confi_weigh[ind] = confi_weigh[ind-1]/2
    
    df.insert(6, "Confi_Weight", confi_weigh)
    
    # Accumulating overall score for each Feed post
    df['Overall'] = df['Time_Weight'] * df['Senti_Value'] * df['Confi_Weight']
    
    # Saving the final table
    df.to_csv('feed_complete.csv')
    
    feed_score = df['Overall'].sum()
    return feed_score

def post_score(s):
    with urlopen(
    "https://graph.facebook.com/v6.0/me?fields=id%2Cname%2Cposts%7Bmessage%2Ccreated_time%7D&access_token={}".format(s)) as response:
        source = response.read()
       
    data = json.loads(source)

    some_data = data['posts']

    outfile = open("post_data.csv", "w")

    fieldNames = ['created_time', 'id', 'message']

    writer = csv.DictWriter(outfile, fieldnames = fieldNames)
    writer.writeheader()

    while len(some_data['data']) != 0:
        with open('post_data.json', 'a+', encoding='utf-8') as f:
            json.dump(some_data['data'], f, ensure_ascii=False, indent=4)
       
        for row in some_data['data']:
            writer.writerow(row)
       
        paging_next = some_data['paging']['next']
       
        with urlopen(paging_next) as response:
            source = response.read()
        some_data = json.loads(source)

    df = pd.read_csv("post_data.csv")

    df = df[pd.notnull(df['message'])]

    df.reset_index(drop = True, inplace=True)
    df.to_csv("post_clean.csv")#, index = False

    senti_value = pd.Series([])
    confidence = pd.Series([])

    for i in range(len(df)):
        temp = []
        column = df['message'][i]
        temp.append(column)
        senti_dict = senti(temp)
        senti_value[i] = Senti_Extractor(senti_dict.get('compound'))
        confidence[i] = senti_dict.get('compound')

    df.insert(3, "Senti_Value", senti_value)
    df.insert(4, "Confidence", confidence)

    # Assigning weightage to DataTime
    total_rows = len(df.index)
    even = False

    time_weigh = pd.Series([])

    if total_rows % 2 == 0:
        even = True

    if even == True:
        i = total_rows/2 - 1
        time_weigh[i] = 25
        i = i-1
        while(i >= 0):
            time_weigh[i] = time_weigh[i+1]/2
            i = i-1
           
        i = total_rows/2
        time_weigh[i] = 25
        i = i+1
        while(i < total_rows):
            time_weigh[i] = time_weigh[i-1]/2
            i = i+1
    else:
        i = total_rows/2 - 0.5
        time_weigh[i] = 50/3
        time_weigh[i+1] = 50/3
        time_weigh[i-1] = 50/3
        time_weigh[i-2] = 25/2
        i = i - 3
        while (i >= 0):
            time_weigh[i] = time_weigh[i+1]/2
            i = i - 1
           
        i = total_rows/2 - 0.5
        time_weigh[i+2] = 25/2
        i = i+3
        while (i < total_rows):
            time_weigh[i] = time_weigh[i-1]/2
            i = i+1
           
    df.insert(2, "Time_Weight", time_weigh)
    df['Confi_Rank'] = df['Confidence'].rank(ascending = 0, method = 'dense')
    df = df.set_index('Confi_Rank')
    df = df.sort_index()

    confi_weigh = pd.Series([])

    confi_weigh[1] = 50

    for ind in df.index:
        if (ind != 1.0):
            confi_weigh[ind] = confi_weigh[ind-1]/2

    df.insert(6, "Confi_Weight", confi_weigh)

    df['Overall'] = df['Time_Weight'] * df['Senti_Value'] * df['Confi_Weight']
    df.to_csv('feed_complete.csv')
    feed_score = df['Overall'].sum()

    return feed_score

def normalize(num):
    while num > 1 or num < -1:
        num = num / 2

    while (num < 0.001 and num > 0) or (num > -0.001 and num < 0):
        num = num * 2

    return num


def Behavoural_analysis(s):
    feed = feed_score(s)
    post = post_score(s)

    feed = normalize(feed)
    post = normalize(post)

    cummilitive_score = (feed + post)/2

    return cummilitive_score
    