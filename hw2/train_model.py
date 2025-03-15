"""
Code for Problem 1 of HW 2.
"""
import pickle
from typing import Any, Dict

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction, AutoModelForSequenceClassification


def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:
    """
    Problem 1d: Implement this function.

    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    :param dataset: A dataset
    :param tokenizer: A tokenizer
    :return: The dataset, prepreprocessed using the tokenizer
    """
    # raise NotImplementedError("Problem 1d has not been completed yet!")
    
    # According to the papaer, the combined length is ≤ 512 tokens.
    
    # The Hugging-Face Transformers fine-tuning API expects datasets to be pre-processed through the following steps.
    # - All input texts should be tokenized.
    # - BERT models have a maximum input length, and all inputs need to be truncated to this length.
    # - Inputs shorter than the maximum input length should be padded to this length.
    # - The pre-processed inputs do not need to be in the form of PyTorch tensors.
    
    MAX_LEN = 512
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN, padding="max_length")
    
    return dataset.map(tokenize_function, batched=True)


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    """
    Problem 2a: Implement this function.

    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: This parameter is required by the Trainer, but it will
        not be used for this problem. Please ignore it
    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be loaded
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A newly initialized pre-trained Transformer classifier
    """
    # raise NotImplementedError("Problem 2a has not been completed yet!")
    
    # Load a pre-trained BERT classifier model from the Hugging Face Model Hub
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # model = BertForSequenceClassification.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # If use_bitfit is True, freeze all non-bias parameters
    if use_bitfit:
        # Search for all parameters in the model
        for name, param in model.named_parameters():
            # If the parameter is not a bias term, freeze it
            if "bias" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    return model


def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """
    # raise NotImplementedError("Problem 2b has not been completed yet!")
    
    # [`Trainer` object](https://huggingface.co/docs/transformers/main_classes/trainer)
    
    # Your `init_trainer` function needs to support the following.
    #- 1.The training configuration (total number of epochs, early stopping criteria if any) must match your answer for Problem 1c.
    #- 2.Your `Trainer` needs to save the model obtained during each training run to a folder called `checkpoints`.
    #- 3.You should leave the `model` keyword parameter blank and instead pass an argument to the `model_init` keyword parameter.
    #- 4.It should evaluate models based on accuracy.
    
    # set checkpoint directory
    if use_bitfit:
        checkpoint_dir = f"train_checkpoints_bitfit"
    else:
        checkpoint_dir = f"train_checkpoints_no_bitfit"
    
    # checkpoint_dir = "checkpoints"
    
    # set training arguments
    TOTAL_EPOCHS = 4
    training_args = TrainingArguments(
        output_dir=checkpoint_dir, # 2.save the model obtained during each training run to a folder called `checkpoints`
        num_train_epochs=TOTAL_EPOCHS, # 1.training configuration match your answer for Problem 1c
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        fp16=True,  # Enable mixed precision training (faster and saves memory)
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # evaluation_strategy="steps", 
        # save_strategy="steps", # 2.save the model obtained during each training run to a folder called `checkpoints`
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        # set early stopping?
    )
    
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
    
    # initialize Trainer
    trainer = Trainer(
        # model=init_model(None, model_name, use_bitfit), # 3.leave the `model` keyword parameter blank
        model_init=lambda trial: init_model(trial, model_name, use_bitfit), # 3.pass an argument to the `model_init` keyword parameter.
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics, # 4.evaluate models based on accuracy
        
    )
    
    return trainer


def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Problem 2c: Implement this function.

    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    # raise NotImplementedError("Problem 2c has not been completed yet!")
    
    # Your code should support the following requirements.
    # - 1.Your hyperparameter tuning configuration must match your answer for Problem 1c.
    # - 2.You must use Optuna for hyperparameter tuning.
    # - 3.You must indicate to Optuna that the hyperparameter search should maximize accuracy.
    
    # batch sizes = 8, 16, 32, 64, 128  
    # learning rates = 3e-4, 1e-4, 5e-5, 3e-5 
    
    # direction=["minimize", "maximize"],
    # backend="optuna",
    # hp_space=optuna_hp_space,
    # n_trials=20,
    # compute_objective=compute_objective,
    
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [3e-4, 1e-4, 5e-5, 3e-5]),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64, 128]),
        }
    
    # search_space = {
    #     "learning_rate": [3e-4, 1e-4, 5e-5, 3e-5],
    #     "per_device_train_batch_size": [8, 16, 32, 64, 128]
    # }
    
    search_space = {
        "learning_rate": [3e-4],
        "per_device_train_batch_size": [8]
    }
    
    def compute_objective(metrics):
        return metrics["eval_accuracy"]
    
    return {
        "direction": "maximize", # 3.indicate to Optuna that the hyperparameter search should maximize accuracy
        "backend": "optuna",
        "hp_space": optuna_hp_space,
        "sampler": optuna.samplers.GridSampler(search_space),
        "compute_objective": compute_objective,
        # "n_trials": 30,
    }


if __name__ == "__main__":  # Use this script to train your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    # # Set up trainer
    # trainer = init_trainer(model_name, imdb["train"], imdb["val"],
    #                        use_bitfit=True)
    # # trainer = init_trainer(model_name, imdb["train"], imdb["val"],
    # #                        use_bitfit=False)

    # # Train and save the best hyperparameters
    # best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    # with open("train_results.p", "wb") as f:
    #     pickle.dump(best, f)
    
    # initialize the Trainer
    trainer_without_bitfit = init_trainer(model_name, imdb["train"], imdb["val"], 
                           use_bitfit=False)
    trainer_with_bitfit = init_trainer(model_name, imdb["train"], imdb["val"], 
                           use_bitfit=True)
    
    # def count_trainable_params(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(f"\nTrainable Parameters WITHOUT BitFit: {count_trainable_params(trainer_without_bitfit.model)}")
    # print(f"\nTrainable Parameters WITH BitFit: {count_trainable_params(trainer_with_bitfit.model)}")
    # print()
    
    # hyperparameter search
    best_without_bitfit = trainer_without_bitfit.hyperparameter_search(**hyperparameter_search_settings())
    best_with_bitfit = trainer_with_bitfit.hyperparameter_search(**hyperparameter_search_settings())
    
    # read the best hyperparameters
    trainer_without_bitfit.args.learning_rate = best_without_bitfit.hyperparameters["learning_rate"]
    trainer_without_bitfit.args.per_device_train_batch_size = best_without_bitfit.hyperparameters["per_device_train_batch_size"]
    
    trainer_with_bitfit.args.learning_rate = best_with_bitfit.hyperparameters["learning_rate"]
    trainer_with_bitfit.args.per_device_train_batch_size = best_with_bitfit.hyperparameters["per_device_train_batch_size"]
    
    # get validation accuracy
    val_acc_without_bitfit = trainer_without_bitfit.evaluate()["eval_accuracy"]
    val_acc_with_bitfit = trainer_with_bitfit.evaluate()["eval_accuracy"]
    
    
    # save the results
    with open("train_results_without_bitfit.p", "wb") as f:
        pickle.dump(best_without_bitfit, f)
    
    with open("train_results_with_bitfit.p", "wb") as f:
        pickle.dump(best_with_bitfit, f)
    
    # print the results
    print("**********************************************************************************")
    print(f"Without BitFit - Validation Accuracy: {val_acc_without_bitfit}, "
      f"Learning Rate: {best_without_bitfit.hyperparameters['learning_rate']}, "
      f"Batch Size: {best_without_bitfit.hyperparameters['per_device_train_batch_size']}")

    print(f"With BitFit - Validation Accuracy: {val_acc_with_bitfit}, "
        f"Learning Rate: {best_with_bitfit.hyperparameters['learning_rate']}, "
        f"Batch Size: {best_with_bitfit.hyperparameters['per_device_train_batch_size']}")
    print("**********************************************************************************")
    
    # save the best checkpoint
    trainer_without_bitfit.save_model("checkpoints/best_model_without_bitfit")
    trainer_with_bitfit.save_model("checkpoints/best_model_with_bitfit")
