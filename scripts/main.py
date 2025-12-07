"""
Main script to run all tasks sequentially.
Usage: python main.py [--task TASK_NUMBER]
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import data_acquisition
from src import segmentation_model
from src import hyperparameter_tuning
from src import refuge_testing
from src import cdr_analysis
from src import config


def run_task1():
    """Run Task 1: Data Acquisition"""
    print("\n" + "="*80)
    print("RUNNING TASK 1: Data Acquisition")
    print("="*80 + "\n")
    
    path = data_acquisition.download_dataset()
    info = data_acquisition.verify_dataset_structure(path)
    
    return info


def run_task2():
    """Run Task 2: Segmentation Model Training"""
    print("\n" + "="*80)
    print("RUNNING TASK 2: Segmentation Model Training")
    print("="*80 + "\n")
    
    model, history = segmentation_model.train_segmentation_model(
        img_dir=config.ORIGA_IMG_DIR,
        mask_dir=config.ORIGA_MASK_DIR,
        num_epochs=config.NUM_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        save_path=config.MODEL_SAVE_PATH
    )
    
    return model, history


def run_task3():
    """Run Task 3: Hyperparameter Tuning"""
    print("\n" + "="*80)
    print("RUNNING TASK 3: Hyperparameter Tuning")
    print("="*80 + "\n")
    
    results = hyperparameter_tuning.run_all_experiments(
        img_dir=config.ORIGA_IMG_DIR,
        mask_dir=config.ORIGA_MASK_DIR,
        num_epochs=config.NUM_EPOCHS
    )
    
    return results


def run_task4():
    """Run Task 4: REFUGE Testing"""
    print("\n" + "="*80)
    print("RUNNING TASK 4: REFUGE Dataset Testing")
    print("="*80 + "\n")
    
    results = refuge_testing.test_on_refuge(
        model_path=config.MODEL_SAVE_PATH,
        refuge_img_dir=config.REFUGE_IMG_DIR,
        refuge_mask_dir=config.REFUGE_MASK_DIR,
        batch_size=config.BATCH_SIZE
    )
    
    return results


def run_task5():
    """Run Task 5: CDR Analysis"""
    print("\n" + "="*80)
    print("RUNNING TASK 5: Cup-to-Disc Ratio Analysis")
    print("="*80 + "\n")
    
    df_results = cdr_analysis.analyze_cdr(
        model_path=config.MODEL_SAVE_PATH,
        refuge_img_dir=config.REFUGE_IMG_DIR,
        refuge_mask_dir=config.REFUGE_MASK_DIR,
        batch_size=1
    )
    
    return df_results


def main():
    """Main function to run selected or all tasks."""
    parser = argparse.ArgumentParser(description='Glaucoma Detection Project - Main Runner')
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4, 5], 
                       help='Run specific task (1-5). If not specified, runs all tasks.')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GLAUCOMA DETECTION PROJECT")
    print("Group 10 - Final Project")
    print("="*80)
    
    if args.task:
        # Run specific task
        if args.task == 1:
            run_task1()
        elif args.task == 2:
            run_task2()
        elif args.task == 3:
            run_task3()
        elif args.task == 4:
            run_task4()
        elif args.task == 5:
            run_task5()
    else:
        # Run all tasks sequentially
        print("\nRunning all tasks sequentially...")
        
        try:
            run_task1()
            run_task2()
            run_task3()
            run_task4()
            run_task5()
            
            print("\n" + "="*80)
            print("ALL TASKS COMPLETED SUCCESSFULLY!")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
