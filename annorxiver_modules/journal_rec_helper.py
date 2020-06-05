import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, KFold

def cross_validation(model, dataset, evaluate, cv=10, random_state=100, **kwargs):
    """
    This function is used to run cross validation on a given model.
    
    arguments:
        model - the model to be evaluated
        dataset - the data for the cross validation procedure
        evaluate - the function to evaluate the given model
        cv - the number of folds to run cv on
        random_state - the seed for reproducability
        kwargs - extra keywords to pass if desired
    """
    
    folds = KFold(n_splits=cv, random_state = random_state, shuffle=True)
    cv_fold_accs = []
    
    fold_predictions = []
    for train, val in folds.split(dataset):
        
        prediction, true_labels = evaluate(
            model, dataset.iloc[train], 
            dataset.iloc[val], **kwargs
        )

        accs = [
                 (
                     1 if true_labels[data_idx] in prediction_row 
                     else 0 
                 )
                 for data_idx, prediction_row in enumerate(prediction)
        ]
        
        cv_fold_accs.append(np.sum(accs)/len(accs))
        print(f"{np.sum(accs)} out of {len(accs)}")
        
        fold_predictions.append(prediction)
        
    print(f"Total Accuracy: {np.mean(cv_fold_accs)*100:.3f}%")
    return fold_predictions

def dummy_evaluate(model, training_data, validation_data, **kwargs):
    """
    This function is used to evaluate a dummy classifier.
    All the classifer does is randomly pick journals for recommendation.
    
    arguments:
        model - dummy classifier
        training_data - the data to train the classifier
        validation_data - the data to evaluate the classifier
        kwargs - extra keywords to pass if desired
    """
    
    top_X = kwargs.get("top_predictions", 10)
    random_states = kwargs.get("dummy_seed", [100,200,300,400,500,600,700,800,900,1000])
    
    X_train = (
        training_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    
    Y_train = (
        training_data
        .journal
        .values
    )
    
    X_val = (
        validation_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    Y_val = (
        validation_data
        .journal
        .values
    )
    
    predictions = []
    for i, seed in zip(range(top_X), random_states):
        model.random_state = seed
        model.fit(X_train, Y_train)
        predictions.append(model.predict(X_val))

    return np.stack(predictions).transpose(), Y_val


def knn_evaluate(model, training_data, validation_data, **kwargs):
    """
    This function is used to evaluate the knearestneighbors classifier 
    on an individual paper by paper basis.
    
    arguments:
        model - knearestneighbor classifier
        training_data - the data to train the classifier
        validation_data - the data to evaluate the classifier
        kwargs - extra keywords to pass if desired
    """
    
    X_train = (
        training_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    
    Y_train = (
        training_data
        .journal
        .values
    )
    
    X_val = (
        validation_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    
    Y_val = (
        validation_data
        .journal
        .values
    )
    
    model.fit(X_train, Y_train)
    distance, neighbors = model.kneighbors(X_val)
    
    predictions = [
        Y_train[neighbor_predict]
        for neighbor_predict in neighbors 
    ]

    return np.stack(predictions), Y_val

def knn_centroid_evaluate(model, training_data, validation_data, **kwargs):
    """
    This function is used to evaluate the knearestneighbors classifier 
    by calculating journal centroids for training.
    
    arguments:
        model - knearestneighbor classifier
        training_data - the data to train the classifier
        validation_data - the data to evaluate the classifier
        kwargs - extra keywords to pass if desired
    """
    
    train_centroid_df = (
        training_data
        .groupby("journal")
        .agg("mean")
        .reset_index()
    )
            
    X_train_centroid = (
        train_centroid_df
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )

    Y_train_centroid = (
        train_centroid_df
        .journal
        .values
    )
    
    
    X_val = (
        validation_data
        .drop("journal", axis=1)
        .values
        .astype('float32')
    )
    
    Y_val = (
        validation_data
        .journal
        .values
    )
    
    model.fit(X_train_centroid, Y_train_centroid)
    distance, neighbors = model.kneighbors(X_val)
    
    predictions = [
        Y_train_centroid[neighbor_predict]
        for neighbor_predict in neighbors 
    ]

    return np.stack(predictions), Y_val