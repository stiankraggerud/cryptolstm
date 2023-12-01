import pandas as pd
import stockDataHandler
data = {
    'Name': ['Anna', 'Bob', 'Charlie', 'David', 'Per'],
    'Age': [25, 30, 35, 40,31],
    'City': ['Oslo', 'Bergen', 'Trondheim', 'Stavanger', "Gokk"]
}

df = pd.DataFrame(data)
name = df['Name']
age = df['Age']
city = df['City']

print(df[df['Age']> 31])

btc = pd.read_csv('training/Nbeats/onehour2022_2003.csv')
btcLoaded = stockDataHandler.LoadData('onehour2022_2003.csv')



