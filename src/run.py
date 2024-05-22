from preprocessing import Preprocessing
from NN import NeuralNetwork
from CNN import CNN
from ensemble import EnsembleModel

def main():
    
    
    data_path = './data/4_Classification of Robots from their conversation sequence_Set2.csv'
    processed_data_path = './data/cleaned_robot_data.csv'
    default_sample_size = 0.1
    skip_preprocessing = False
    skip_NN = False
    skip_Ensemple = True
    Test_CNN = True
    
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
        print("\nData Preprocessing Complete.")
    
    if skip_NN:
        print("\nNeural Network training skipped.")
    elif Test_CNN != True:
        print("\nNeural Network training being processed...")
        # Neural Network training and evaluation
        nn = NeuralNetwork(data_path, processed_data_path)
        nn.load_and_preprocess_data()
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

    if skip_Ensemple:
        print("\nEnsemble model training skipped.")
    else:
        print("\nEnsemble model training being processed...")
        #Ensemble model training and evaluation
        ensemble = EnsembleModel(data_path, processed_data_path)
        ensemble.train_and_evaluate()
        print("\Ensemble model complete.")  
        
    print("\nData preprocessing and model training complete.")   

if __name__ == "__main__":
    main()


