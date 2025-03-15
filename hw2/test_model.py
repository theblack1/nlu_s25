"""
Code for Problem 1 of HW 2.
"""
import pickle

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset

import numpy as np
from transformers import EvalPrediction
import os
import json


def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    # raise NotImplementedError("Problem 2b has not been completed yet!")
    
    # [`Trainer` object](https://huggingface.co/docs/transformers/main_classes/trainer)
    # Your `init_tester` function needs to support the following.
    # - 1.The `Trainer` should only support testing and not training.
    # - 2.It should evaluate models based on accuracy.
    
    # # load the accuracy metric
    # metric = evaluate.load("accuracy")
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return metric.compute(predictions=predictions, references=labels)
    
    # load the accuracy metric
    metric = evaluate.load("accuracy")
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return metric.compute(predictions=predictions, references=labels)
    
    def compute_metrics(eval_pred: EvalPrediction):
        logits, labels = eval_pred.predictions, eval_pred.label_ids
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # set testing arguments
    testing_args = TrainingArguments(
        output_dir="test_checkpoints", 
        do_train=False, # 1. Trainer should only support testing and not training
        do_predict=True, # 1. Trainer should only support testing and not training
        evaluation_strategy="no", # 1. Trainer should only support testing and not training
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=True,  # Enable mixed precision training (faster and saves memory)
    )

    tester = Trainer(
        model=BertForSequenceClassification.from_pretrained(directory),
        args=testing_args,
        compute_metrics=compute_metrics,
    )
    
    return tester

# # auto find the best checkpoint based on `eval_accuracy`
# def get_best_checkpoint_from_logs(directory):
#     """
#     递归遍历 `directory` 及其子目录，找到 `eval_accuracy` 最高的 checkpoint。
    
#     :param directory: 训练 checkpoints 目录，例如 "./train_checkpoints_no_bitfit"
#     :return: (最佳 checkpoint 路径, 最佳 eval_accuracy)
#     """
#     best_checkpoint = None
#     best_accuracy = 0.0
#     best_step = 0
    
#     # traverse the directory and all subdirectories
#     for root, _, files in os.walk(directory):
#         if "trainer_state.json" in files:
#             trainer_state_path = os.path.join(root, "trainer_state.json")

#             # read the trainer_state.json
#             with open(trainer_state_path, "r") as f:
#                 trainer_state = json.load(f)

#             # traverse log_history to find the highest eval_accuracy step
#             for log in trainer_state.get("log_history", []):
#                 if "eval_accuracy" in log:  # ensure that there is an eval_accuracy record
#                     accuracy = log["eval_accuracy"]
#                     step = log["step"]

#                     if accuracy > best_accuracy:
#                         best_accuracy = accuracy
#                         best_step = step
#                         best_checkpoint = root

#     print(f"Best checkpoint in {directory}: {best_checkpoint}, Best eval_accuracy: {best_accuracy}, Best step: {best_step}")
#     print()
    
#     return best_checkpoint


if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # # Set up tester
    # tester = init_tester("./train_results_with_bitfit.p")

    # # Test
    # results = tester.predict(imdb["test"])
    # with open("test_results.p", "wb") as f:
    #     pickle.dump(results, f)

    # find the best checkpoint
    best_checkpoint_no_bitfit = "checkpoints/best_model_without_bitfit"
    best_checkpoint_bitfit = "checkpoints/best_model_with_bitfit"
    
    # test WITHOUT BitFit
    tester_without_bitfit = init_tester(best_checkpoint_no_bitfit)
    results_without_bitfit = tester_without_bitfit.predict(imdb["test"])
    
    with open("test_results_without_bitfit.p", "wb") as f:
        pickle.dump(results_without_bitfit, f)

    # test WITH BitFit
    tester_with_bitfit = init_tester(best_checkpoint_bitfit)
    
    for name, param in tester_with_bitfit.model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    results_with_bitfit = tester_with_bitfit.predict(imdb["test"])
    
    with open("test_results_with_bitfit.p", "wb") as f:
        pickle.dump(results_with_bitfit, f)
    
    # Function to count trainable parameters
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get Trainable Parameters
    trainable_params_without_bitfit = count_trainable_params(tester_without_bitfit.model)
    trainable_params_with_bitfit = count_trainable_params(tester_with_bitfit.model)

    # Get Test Accuracy
    test_acc_without_bitfit = results_without_bitfit.metrics["test_accuracy"]
    test_acc_with_bitfit = results_with_bitfit.metrics["test_accuracy"]

    print("**********************************************************************************")
    print(f"Without BitFit - Trainable Parameters: {trainable_params_without_bitfit}, Test Accuracy: {test_acc_without_bitfit}")
    print(f"With BitFit - Trainable Parameters: {trainable_params_with_bitfit}, Test Accuracy: {test_acc_with_bitfit}")
    print("**********************************************************************************")