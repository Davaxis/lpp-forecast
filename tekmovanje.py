from calendar import weekday
from datetime import datetime, time
import numpy as np
import pandas as pd
import linear
import lpputils

route_dict = {}
direction_dict = {}

def read_data(filename, sep='\t'):
    """
    Reads the data from the given file.
    """
    return pd.read_csv(filename, sep=sep)

def add_day_of_week(data):
    """
    Adds the day of the week to the data represented as int.
    """
    rows = pd.to_datetime(data['Departure time'])
    days = np.zeros((len(rows), 7))
    for index, row in enumerate(rows):
        days[index, row.weekday()] = 1
    days = pd.DataFrame(days, columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    data = pd.concat([data, days], axis=1)
    return data

def add_holiday_info(data):
    """
    Adds information if a day is holiday to the data.
    """
    holidays = read_data('prazniki.csv', ';')
    dates = list(holidays['DATUM'])
    data['Holiday'] = data['Departure time'].apply(lambda x: 1 if datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f').strftime('%-d.%m.%Y') in dates else 0)
    return data

def add_duration(data):
    """
    Adds the duration of the trip to the data.
    """
    data['Duration'] = data.apply(lambda x: lpputils.tsdiff(x['Arrival time'], x['Departure time']), axis=1)
    return data

def add_route(data, no_of_routes=30):
    """
    Adds the information of which route is the bus driving on to the data.
    """
    routes = data['Route']
    output = np.zeros((len(routes), no_of_routes))
    dict_index = 0
    for row_index, route in enumerate(routes):
        if route in route_dict:
            output[row_index, route_dict[route]] = 1
        else:
            route_dict[route] = dict_index
            output[row_index, dict_index] = 1
            dict_index += 1
    output = pd.DataFrame(output)
    data = pd.concat([data, output], axis=1)
    return data

def get_direction_from_row(row):
    direction = row['Route description']
    route = row['Route']
    if route in direction_dict:
        if direction == direction_dict[route]:
            return 0
        else:
            return 1

    direction_dict[route] = direction
    return 0

def add_direction(data):
    """
    Adds information about in which direction the bus is driving to the data.
    """
    data['Direction'] = data.apply(get_direction_from_row, axis=1)
    return data

def pre_process_data(data, train=True):
    """
    Pre-processes the data.
    """

    data = add_day_of_week(data)
    data = add_holiday_info(data)
    data = add_route(data)
    data = add_direction(data)

    # not really needed since they are the same everywhere
    data = data.drop('Route description', axis=1)
    data = data.drop('Route Direction', axis=1)
    data = data.drop('First station' , axis=1)
    data = data.drop('Last station', axis=1)
    data = data.drop('Route', axis=1)

    data = data.drop('Registration', axis=1)
    data = data.drop('Driver ID', axis=1)

    if train:
        add_duration(data)
    data = data.drop('Arrival time', axis=1)

    departures = data['Departure time']
    data = data.drop('Departure time', axis=1)

    # print(data)

    return data, departures

def train_lr(data, lamb=0.5, label='Duration'):
    """
    Trains the linear regression model.
    """
    X = data.drop(label, axis=1).to_numpy()
    y = data[label].to_numpy()

    lr = linear.LinearLearner(lambda_=lamb)
    return lr(X,y)

def predict_lr(model, data):
    """
    Predicts the arrival time for the given data. Data should be pre-processed.
    """
    rows = data.to_numpy()
    results = []
    for row in rows:
        results.append(model(row))

    data['Duration'] = results
    return data

def post_process(data, departures):
    """
    Post-processes the data.
    """
    data['Departure time'] = departures
    data['Arrival time'] = data.apply(lambda x: lpputils.tsadd(x['Departure time'], x['Duration']), axis=1)
    return data

def create_output(data, departures, filename='out.txt'):
    """
    Creates the output file.
    """
    data = post_process(data, departures)
    data['Arrival time'].to_csv(filename, sep='\n', index=False, header=False)

if __name__ == '__main__':
    train_data = read_data('train.csv')
    test_data = read_data('test.csv')
    train_data, departures_train = pre_process_data(train_data)
    test_data, departures_test = pre_process_data(test_data, train=False)
    model = train_lr(train_data)
    pred = predict_lr(model, test_data)
    create_output(pred, departures_test)
    