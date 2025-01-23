from data_preprocessing import DataPreprocessor, validate_tokenized_data
from config import TrainingConfig

def prepare_and_save_data(config: TrainingConfig):
    """Prepare and save training and validation datasets"""
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config.model_name, config)
    
    # Load and verify data
    data = preprocessor.load_and_verify_data(config.data_path)
    
    # Create and split datasets
    train_dataset, val_dataset = preprocessor.prepare_data(data, config.train_ratio)
    
    # Validate datasets
    print("\n🔎 Validating datasets...")
    validate_tokenized_data(train_dataset)
    validate_tokenized_data(val_dataset)
    
    # Save datasets
    preprocessor.save_datasets(
        train_dataset, 
        val_dataset,
        config.train_dataset_path,
        config.val_dataset_path
    )

def main():
    """Main function to prepare and save datasets."""
    config = TrainingConfig()
    prepare_and_save_data(config)

# Prevents main from running as an import
if __name__ == "__main__":
    main()