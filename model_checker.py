"""
This file defines exactly how we will validate your models. Please carefully
look through what is present here and make sure you understand what each
function does.

NOTE: If your submitted code fails any of the requirements in this file, we
will have to contact you to resubmit. If there is no time, then you will be
automatically disqualified. So please make sure this code file runs perfectly
and read the outputs before submitting.
"""
# IMPORT YOUR MODULES HERE ----------------------------------------------------
import pandas as pd
import numpy as np
import pickle
# -----------------------------------------------------------------------------

# DO NOT ALTER BEYOND THIS POINT ----------------------------------------------

import code  # importing your code.py module
from code import saved_model_filename  # Loading your model name


def profit_checker(data_year1, data_year2, loaded_model):
    """
    (DO NOT ALTER THIS FUNCTION)

    This function checks whether your model makes money on both years of the
    training data.

    Inputs:
        - data_year1: A pandas dataframe. This is the training data for year 1
        - data_year2: A pandas dataframe. This is the training data for year 2
        - loaded_model: This is the result of your train() function that was
                        saved

    Outputs:
        - makes_profit: A boolean (True or False). Indicates whether your model
                        makes a profit on training data from both year 1 and
                        year 2
    """
    year1_loss = data_year1['AMOUNT'].sum()
    year2_loss = data_year2['AMOUNT'].sum()

    year1_revenue, year2_revenue = 0, 0
    for ind, row in data_year1.iterrows():
        row = np.asarray(row)
        year1_revenue += code.predict(row, loaded_model)
        year2_revenue += code.predict(row, loaded_model)

    year1_profit = year1_revenue - year1_loss
    year2_profit = year2_revenue - year2_loss

    makes_profit = (year1_profit > 0) and (year2_profit > 0)
    makes_profit = np.prod(makes_profit)
    makes_profit = bool(makes_profit)
    return makes_profit


def validate_model(loaded_model, fake_year3):
    """
    (DO NOT ALTER THIS FUNCTION)

    This function ensures that your predict function does indeed create the
    validation_year3.csv file that was created by using your saved model.

    Inputs:
        - loaded_model: This is the result of load_model() from code.py
        - fake_year3: A pandas dataframe containing randomly generated fake
                      data.

    Outputs:
        - macth: A boolean value (True or False) indicating whether the
                 validation_year3 file matches what is created by your predict
                 function.
    """
    model_premiums = pd.DataFrame()
    model_premiums['IDCNT'] = fake_year3['IDCNT'].copy()
    model_premiums['PREMIUM'] = 0

    premium_array = []
    for ind, row in fake_year3.iterrows():
        row = np.asarray(row)
        row_premium = code.predict(row, loaded_model)
        premium_array.append(row_premium)

    model_premiums['PREMIUM'] = premium_array

    val_year3 = pd.read_csv('validation_year3.csv')
    match = abs(model_premiums['PREMIUM'] - val_year3['PREMIUM']) < 1e-10
    match = np.prod(match)
    match = bool(match)
    return match


def model_checker():
    """
    (DO NOT ALTER THIS FUNCTION)

    This function checks that your model is valid by running the
    validate_model() function.

    It then checks whether your model is profitable on training data.

    If either of the above two fails it will raise an assertion error. 
    """

    print('Loading the data')
    data_year1 = pd.read_csv('data_year1.csv')
    data_year2 = pd.read_csv('data_year2.csv')
    fake_year3 = pd.read_csv('fake_year3.csv')

    loaded_model = code.load_model(saved_model_filename)

    profit = profit_checker(data_year1, data_year2, loaded_model)
    valid = validate_model(loaded_model, fake_year3)

    print('\nIs your model reproducible?                  {0:s}'
          .format(str(valid)))
    print('Does your model make money on training data? {0:s}'.
          format(str(profit)))
    print('Can you upload your model as is?             {0:s}'.
          format(str(profit and valid)))


# Finally the model_checker() is executed
if __name__ == '__main__':
    model_checker()
