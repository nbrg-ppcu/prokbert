from prokbert.curriculum_utils import get_embedding, evaluate_embeddings
from transformers import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AdamW

import torch

class CustomTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            # Separate parameters
            bert_params = [p for n, p in self.model.named_parameters() if "model" in n] #bert
            head_params = [p for n, p in self.model.named_parameters() if "model" not in n]
            
            self.optimizer = AdamW([
                {"params": bert_params, "lr": self.args.backbone_lr_rate},
                {"params": head_params, "lr": self.args.head_lr_rate}
            ],
            betas=(self.args.beta_1, self.args.beta_2))
        return self.optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        optimizer = optimizer or self.optimizer
        self.lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",    
            factor=0.5,  
            patience=5,
            verbose=True
        )
        return self.lr_scheduler

    def train(self, *args, **kwargs):
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(num_training_steps=0, optimizer=self.optimizer)
        return super().train(*args, **kwargs)

    def evaluate(self, eval_dataset=None, **kwargs):
        eval_dataset = eval_dataset or self.eval_dataset
        result = super().evaluate(eval_dataset=None, **kwargs)
        emb_dataset = get_embedding(self.model, eval_dataset, self.data_collator, self.model.device)
        score = evaluate_embeddings(emb_dataset, eval_dataset["labels"])
        metrics = {"eval_silhouette_score":  score}
        result["eval_silhouette_score"] = score
        self.log(metrics)
        return result

    def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics)

        # Run delayed LR scheduler now that metrics are populated
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and not skip_scheduler:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                self.lr_scheduler.step(metrics[metric_to_check])
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', "
                    f"which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. "
                    f"Please ensure that the `compute_metrics` function returns a dictionary that includes '{metric_to_check}' or "
                    f"consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc
        return metrics