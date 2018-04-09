"""
NOTE: Please make sure that all of the files related to the pricing games are
in one folder.

This file has 4 sections that you should fill out:

    Section 1: Place code that trains your model
    Section 2: Creating the functions that save and load your model
               (e.g. model.pickle)
    Section 3: Fill in the example predict() function with your own code
    Section 4: Generate the validation_year3.csv file
"""

###############################################################################
################################## SECTION 1 ##################################
# In this section you should put all the code that you will need to train your
# model
#
# In the toy example below mu model will take the total loss in years 1 and 2
# and set the pure actuarial premium to be the average loss per person of the
# year where the average loss per person was the largest.
#
# I will then (in the section 2) add to the premium price the remainder of the
# unique contract ID (IDCNT), when divided by 42. The reason for this is for
# my toy model to have some row dependent feature. Nothing more.


# Importing all the modules that I need for my model which is only pandas

# IMPORT YOUR MODULES HERE ---------------------------------------------------
import pandas as pd
import numpy as np
# ----------------------------------------------------------------------------


def train():
    """
    This function trains your model and returns the trained model.
    This function should return all the information that is gained by training
    your model (e.g. parameter values, the model itself, etc). Note that you
    can also return a list or a tuple or any obejct.
    """

    data_year1 = pd.read_csv('data_year1.csv')  # Reading the data from year 1
    data_year2 = pd.read_csv('data_year2.csv')  # Reading the data from year 2

    # YOUR CODE STARTS HERE ---------------------------------------------------

    # Computing the total loss for years 1 and 2
    year1_loss = data_year1['AMOUNT'].sum()
    year2_loss = data_year2['AMOUNT'].sum()

    # Computing the post-hoc pure actuarial premiums which is the per capita loss
    year1_pure_premium = year1_loss / len(data_year1)
    year2_pure_premium = year2_loss / len(data_year2)

    # Finding the larger of the two and setting it as the pure premium
    train_result = max(year1_pure_premium, year2_pure_premium)

    return train_result


###############################################################################
################################## SECTION 2 ##################################
# In this section you should save the output of your train() function.
# There are two functions, one saves your model and the other loads it.


# In the toy example I use python3's built in pickle

# YOU SHOULD IMPORT THE MODULES THAT YOU NEED FOR SAVING AND LOADING IF ANY
import pickle


def save_model(train_result, model_file_name):
    """
    This function will take as input the result of what was returned by train()
    and a file name. It will then save the training result.
    It will then save the train_result in a file named model_file_name.

    Inputs:
        - train_result: This is what is returned by the train() function
        - model_file_name: This is a string which labels the object to be
                           saved (e.g. 'model.pickle')

    In the toy example below I use pickle to save my pure_premium value (see
    above).
    """
    # YOUR CODE HERE ----------------------------------------------------------

    # Comment out the next 2 lines when you are writing your own save function
    with open(model_file_name, 'wb') as f:
        pickle.dump(train_result, f)

    pass


def load_model(saved_model_filename):
    """
    This function loads whatever was saved by save_model().

    Inputs:
        - saved_model_filename: a string with the full file name
                                (e.g. 'model.pickle').

    Outputs:
        - model: the loaded object.


    In the toy example I load a pickle file.
    """
    # YOUR CODE HERE ---------------------------------------------------------

    # Comment out the next 2 lines when you are writing your own load function
    with open(saved_model_filename, 'rb') as f:
        model = pickle.load(f)

    return model


###############################################################################
################################## SECTION 3 ##################################
# In this section you should alter the contents of the predict function.
#
# In the toy example, I divide the contract ID (IDCNT) of each contract by 42
# and add the remainder to the pure_premium value I have stored in my pickled
# object. The reason for this is for my toy model to have some row dependent
# feature. Nothing more.


def predict(row, saved_object):
    """
    This takes in a SINGLE ROW of data and a saved_object and outputs
    a premium price as a floating point number.

    You should edit this function so that it works with your saved_object
    and your model.

    Inputs:

        - row: This is a numpy array which represents a single row of the
               training data.

        - saved_model: This is your saved model as done by save_model().

    Output:

        - premium: This is a python float and represents the premium offered
                   by your model.
    """

    # YOUR CODE HERE -------------------------------------------------------
    # Make sure to define a final "premium" variable

    # Comment out the next 3 lines when you start building your model
    pure_premium = saved_object  # My pure premium is the only element

    contract_id = row[0]  # Contract ID is the first element

    premium = float(pure_premium + (contract_id % 42))  # Adding the remainder

    return premium


###############################################################################
################################## SECTION 4 ##################################
# This section trains your model, saves it, then loads your model and
# generates a validation_year3.csv file. You should only change the value of
# saved_model_filename the appropriate name to save your model in.


# CHANGE THE VALUE OF THIS VARIABLE
saved_model_filename = 'model.pickle'


# DO NOT ALTER BEYOND THIS POINT ----------------------------------------------
if __name__ == '__main__':
    # Loading the data
    fake_year3 = pd.read_csv('fake_year3.csv')

    # Creating an empty dataframe with just two columns
    #
    #    1. IDCNT: These are the contract IDs
    #    2. PREMIUM: These will be the premiums offered by your model
    val_year3 = pd.DataFrame()
    val_year3['IDCNT'] = fake_year3['IDCNT'].copy()
    val_year3['PREMIUM'] = 0  # Initially setting them to zero

    # First we train the model
    model = train()

    # Next we save the model
    save_model(model, saved_model_filename)

    # Next we load the model
    model = load_model(saved_model_filename)

    # Next we will compute the premiums using the predict function.
    premium_array = []  # A list that will contain the premium values
    for ind, row in fake_year3.iterrows():
        row = np.asarray(row)
        row_premium = predict(row, model)  # start building your model
        premium_array.append(row_premium)

    val_year3['PREMIUM'] = premium_array  # setting the premium values

    # Saving the data as a csv file
    val_year3.to_csv('validation_year3.csv', index=False)
