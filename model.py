from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, pipeline
from transformers import TrainingArguments, Trainer
from transformers import pipeline

from preprocessing import generate_train_data


class BertBaseLine:
    def __init__(self):
        self.trainer = None
        self.data_collator = None
        self.tokenizer = None
        self.model = None
        self.model_name = "distilbert-NER"
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.eval_metrics = None

    def pretrained_model(self):
        """
            downloads pretrained model
        :return:
            model
        """
        self.tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)

    def prepare_dataset(self, training_files, testing_files):
        """
            calls preprocessor of the data to prepare the dataset which can be sent to
            the pretrained model for finetuning
        :return:
            train, validation, test datasets
        """
        train_token_list, train_input_ids, train_input_masks, train_segment_ids, train_labels = generate_train_data(
            training_files, 20, self.tokenizer)

    def compute_metrics(self):
        """
            evaluation metrics for the model
        :return:
        """

    def prepare_trainer(self):
        """
            prepared the training arguments to retrain the model with our data set
        :return:
            trainer
        """
        training_args = TrainingArguments(
            output_dir="bert-baseline",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        self.compute_metrics()

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.eval_metrics,
        )

    def train(self):
        """
            initializes the model and start training. Also stores the trained model
        :return:
            None
        """
        self.pretrained_model()
        self.prepare_trainer()
        self.trainer.train()

        self.model.save_pretrained("bert-baseline")
        self.tokenizer.save_pretrained("tokenizer-bert-baseline")

    def test_performance(self):
        """
            calls evaluation on test data to get the accuracy
        :return:
            accuracy if the model for test set
        """
        eval_results = self.trainer.evaluate(self.test_dataset)
        print("Test Set Accuracy:", eval_results["eval_accuracy"])

        eval_results = self.trainer.evaluate(self.validation_dataset)
        print("Validation Set Accuracy:", eval_results["eval_accuracy"])

    def get_samples(self):
        """
            provides a list of example sentences on which the model can be tested
        :return:
            list of samples
        """
        samples = []
        sample1 = "Dr.Dilip checked the patient shiva on 08-11-2024"
        sample2 = "Dilip is a well known pulmonary doctor in Pittsburgh"
        samples.append(sample1)
        samples.append(sample2)
        return samples

    def run_samples(self):
        """
            give the sample sentences to get the deidentified text
        :return:
            deidentified sentence
        """
        pipe = pipeline('token-classification', model='bert-baseline', tokenizer=self.tokenizer
                        , device='cuda', trust_remote_code=True)

        examples = self.get_samples()
        for sample in examples:
            print(pipe(sample))
