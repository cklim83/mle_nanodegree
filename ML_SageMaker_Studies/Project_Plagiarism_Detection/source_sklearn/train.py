from __future__ import print_function

import argparse
import os
import pandas as pd

from sklearn.externals import joblib

## TODO: Import any additional libraries you need to define a model
from sklearn.ensemble import RandomForestClassifier


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--n_estimators', type=int, default=100, metavar='N',
                        help='number of estimators in random forest (default: 100')
    parser.add_argument('--max_depth', type=int, default=6, 
                        help='max depth of each tree (default: 6)')
    parser.add_argument('--max_features', type=float,  default=0.5,
                        help='fraction of total features to consider for each split (default: 0.5)')
    parser.add_argument('--oob_score', type=bool, default=True,
                        help='to retain the out of bag score during training (default: True)')
    parser.add_argument('--random_state', type=int, default=5,
                        help='random seed for repeatability of result (default: 5')
                      
        
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    

    ## TODO: Define a model 
    model = RandomForestClassifier(n_estimators=args.n_estimators,
                                   max_depth=args.max_depth,
                                   max_features=args.max_features,
                                   oob_score=args.oob_score,
                                   random_state=args.random_state)
    
    
    ## TODO: Train the model
    model.fit(train_x, train_y)
    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))