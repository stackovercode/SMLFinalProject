from preprocessing import Preprocessing
from NN import NeuralNetwork
from CNN import CNN
from ensemble import EnsembleModel
from GB import GradientBoosting
from compare import ModelComparison

def main():
    data_path = './data/4_Classification of Robots from their conversation sequence_Set2.csv'
    processed_data_path = './data/cleaned_robot_data.csv'
    #default_sample_size = 0.1
    default_sample_size = 0.4
    skip_preprocessing = False
    skip_gb = False
    skip_NN = False
    skip_Ensemple = True
    Test_CNN = False
    
    if skip_preprocessing:
        print("\nData has already been preprocessed. Skipping preprocessing step.")
    else:
        # Preprocessing step
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
    
    if skip_NN:
        print("\nNeural Network training skipped.")
    elif Test_CNN != True:
        print("\nNeural Network training being processed...")
        # Neural Network training and evaluation
        nn = NeuralNetwork()
        nn.load_data(X_train, X_test, y_train, y_test)
        nn.build_model()
        nn.train_model()
        nn.evaluate_model()
        nn.plot_training_history()
        print("\nNeural Network model complete.")
    else:
        print("\nNeural Network training skipped")
        print("\nCNN model training being processed...")
        # CNN model training and evaluation
        cnn = CNN(data_path, processed_data_path)
        cnn.load_and_preprocess_data()
        cnn.build_model()
        cnn.train_model()
        cnn.evaluate_model()
        cnn.plot_training_history()
        print("\nCNN model complete.")
        
    if skip_gb:
        print("\nGradient Boosting training skipped.")
    else:
        print("\nGradient Boosting training being processed...")
        # Gradient Boosting training and evaluation
        gb = GradientBoosting()
        gb.load_data(X_train, X_test, y_train, y_test)
        gb.build_model()
        gb.train_model()
        gb.evaluate_model()
        print("\nGradient Boosting model complete.")


    if skip_Ensemple:
        print("\nEnsemble model training skipped.")
    else:
        print("\nEnsemble model training being processed...")
        #Ensemble model training and evaluation
        ensemble = EnsembleModel(data_path, processed_data_path)
        ensemble.train_and_evaluate()
        print("\Ensemble model complete.")  
        
    # Run comparison after both models are trained and evaluated
    comparator = ModelComparison()
    comparator.run_comparison()
    print("\nComparison of models complete.")
        
    print("\nData preprocessing and model training complete.")   

if __name__ == "__main__":
    main()


