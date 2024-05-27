from preprocessing import Preprocessing
from NN import NeuralNetwork
from GB import GradientBoosting
from compare import ModelComparison

def main():
    
    data_path = './data/4_Classification of Robots from their conversation sequence_Set2.csv'
    processed_data_path = './data/cleaned_robot_data.csv'
    default_sample_size = 0.1
    
    ########## Parameters Neural Network ##########
    skip_NN = False
    skip_validate_NN_params = False
    
    ########## Parameters Gradient Boosting ##########
    skip_gb = True
    skip_validate_GB_params = True

    ########## Data Preprocessing ##########
    print("\nData Preprocessing Being processed...")
    preprocessor = Preprocessing(data_path, processed_data_path, default_sample_size)
    
    sample_fracs = {
        0: default_sample_size,
        1: default_sample_size,
        2: default_sample_size,
        3: default_sample_size,
        4: default_sample_size
    }

    preprocessor.load_data(sample_fracs)
    preprocessor.save_cleaned_data()
    
    total_robots = len(preprocessor.df_sample['source'].unique())
    for i, robot_id in enumerate(preprocessor.df_sample['source'].unique()):
        preprocessor.examine_robot_data(robot_id, i + 1, total_robots)

    preprocessor.plot_correlation_heatmap(preprocessor.df_sample.drop('source', axis=1))

    # Apply PCA during preprocessing
    X_train, X_test, y_train, y_test, explained_variance_ratio = preprocessor.preprocess_data(apply_pca=True)
    print("\nData Preprocessing Complete.")

    ##########################################
    ########## Neural Network ###############
    ##########################################
    
    if skip_NN:
        print("\nNeural Network training skipped.")
    else:
        print("\nNeural Network training being processed...")
        nn = NeuralNetwork()
        nn.load_data(X_train, X_test, y_train, y_test)
        if skip_validate_NN_params!=True:
            nn.hyperparameter_tuning()
        else:
            best_params_NN = {
            'model__activation': 'relu',
            'model__optimizer': 'adam',
            'model__dropout_rate': 0.2,
            'batch_size': 32,
            'epochs': 100
            }
            nn.set_params(best_params_NN)
        nn.train_model()
        nn.evaluate_model()
        nn.plot_training_history()
        print("\nNeural Network model complete.")
        
    if skip_gb:
        print("\nGradient Boosting training skipped.")
    else:
        print("\nGradient Boosting training being processed...")
        gb = GradientBoosting()
        gb.load_data(X_train, X_test, y_train, y_test)
        if skip_validate_GB_params!=True:
                gb.hyperparameter_tuning()
        else:
            best_params_GB = {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 3,
                'subsample': 1.0,
                'min_samples_split': 2
            }
            gb.set_params(best_params_GB)
        gb.train_model()
        gb.evaluate_model()
        feature_importances = gb.get_feature_importance()
        print("Feature Importances:", feature_importances)
        gb.plot_feature_importance()
        print("\nGradient Boosting model complete.")
        
    ##########################################
    ########## Comparison of models ##########
    ##########################################
    comparator = ModelComparison()
    comparator.run_comparison()
    print("\nComparison of models complete.")
        
    print("\nData preprocessing and model training complete.")   

if __name__ == "__main__":
    main()


