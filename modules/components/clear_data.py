def clear_data(dataFrame):
    print('Clearing data...')
    df = dataFrame.drop('id', 1)
    print('Clearing finished!')

    return df
