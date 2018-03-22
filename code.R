###############################################################################
# NOTE: Please make sure that all of the files related to the pricing games are
# in one folder.
#
# This file has 4 sections that you should fill out:
#
#    Section 1: Place code that trains your model
#    Section 2: Fill in the model_pricing() function with your own code
#    Section 3: Generate the validation_year3.csv file
###############################################################################

###############################################################################
################################## SECTION 1 ##################################
# In this section you should put all the code that you will need to train your
# model
#
# The skeleton model I will build will take the total loss in years 1 and 2
# and set the pure actuarial premium to be the average loss per person of the
# year where the average loss per person was the largest.
#
# I will then add to the premium price the remainder of the unique contract ID
# IDCNT, when divided by 42.

train = function(){
    # This function trains your model and returns the trained model.
    # This model returns an object which is saved in trained_model.RData

    data_year1 = read.csv('data_year1.csv')  # Reading the data from year 1
    data_year2 = read.csv('data_year2.csv')  # Reading the data from year 2

    # YOUR CODE HERE ------------------------------------------------------

    # Computing the total loss for years 1 and 2
    year1_loss = sum(data_year1[,'AMOUNT'])
    year2_loss = sum(data_year2[,'AMOUNT'])
    
    # Computing the post-hoc pure actuarial premiums which is the per capita loss
    year1_pure_premium = year1_loss / nrow(data_year1)
    year2_pure_premium = year2_loss / nrow(data_year2)

    # Finding the larger of the two and setting it as the pure premium
    trained_model = max(year1_pure_premium, year2_pure_premium)

    # ---------------------------------------------------------------------
    # The result trained_model is something that you will save in the next section
    return(trained_model)
}


###############################################################################
################################## SECTION 2 ##################################
# In this section you should edit the predict_premium function below so that it
# works with any dataset with variables as in 'data_year1.csv' and 
# 'data_year2.csv' and that gives premium prices for any row of the data frame.
#
# In the toy example, I need only divide the contract ID (IDCNT) of each
# contract by 42 and add the remainder to the pure_premium value I have computed
# previously

predict_premium = function(row, trained_model){
        
    #   This takes in a single row of the data and outputs a premium price.
    #
    #   You should edit this function so that it works with your sa
    #   and your model.
    #
    #   Inputs:
    #
    #       - row: This is a single row from the data.
    #       - trained_model: This is the result of the train() function that
    #                       you will have saved.
    #
    #   Output:
    #
    #       - premium: This is a number and represents the premium offered by
    #                  your model for this row.

    # YOUR CODE HERE ------------------------------------------------------

    pure_premium = trained_model  # Loaded result
    
    contract_id = as.integer(row[[1]])  # We know the first element is the IDCNT
    remainder = contract_id %% 42
    premium = pure_premium + remainder  # pricing formula

    return(premium)
    # ---------------------------------------------------------------------
}

###############################################################################
################################## SECTION 3 ##################################
# This section trains your model, saves it into a file labeled "model.RData",
# then loads your model and generates a validation_year3.csv file. 

# DO NOT ALTER BEYOND THIS POINT ----------------------------------------------

generate_validation = function(){
  
  # Loading the data
  fake_year3 = read.csv('fake_year3.csv')
  
  # Creating an empty dataframe with just two columns
  #
  #    1. IDCNT: These are the contract IDs
  #    2. PREMIUM: These will be the premiums offered by your model
  val_year3 = data.frame(IDCNT = fake_year3[,'IDCNT'], PREMIUM = 0)
  
  # Training your model
  trained_model = train()
  
  # Saving your model (DO NOT ALTER)
  save(trained_model, file='trained_model.RData')
  
  trained_model = NULL # To make sure that we load the result from file
  
  # Loading your model
  load(file='trained_model.RData')
  
  # Making row-wise predictions on the fake_year3.csv
  premiums = apply(fake_year3, 1, predict_premium, trained_model=trained_model)
  
  # Setting the premium predictions on fake data in validation_year3.csv
  val_year3['PREMIUM'] = premiums
  
  # Saving the data as a csv file
  write.csv(x = val_year3, file = "validation_year3.csv", row.names = FALSE)
  
}

# This code will only run during training
if (interactive()) {
  generate_validation()
}

# commentaire
