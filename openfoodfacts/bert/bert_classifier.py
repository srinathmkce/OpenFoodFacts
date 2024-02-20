import torch
import lightning as L
from torch import nn
from transformers import BertModel
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


class BertClassifier(L.LightningModule):
    def __init__(self, dataset, lr):
        """
        Initializes the network, optimizer and scheduler
        """
        super().__init__()
        self.dataset = dataset
        self.lr = lr
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.bert_model = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.drop = nn.Dropout(p=0.2)
        # assigning labels
        self.class_names = [
            "Organic Beverages and Snacks",
            "Cheese and Bread Products",
            "Chocolate, Fruits, and Cheese",
            "International Food Items",
            "Caloric Content and Nutritional Information",
            "Integral and Supplemental Foods",
            "Dietary Supplements and Complements",
            "Energy and Ultra-Processed Products",
            "Halal and Dietary Restrictions",
            "Beverages and Instant Drinks",
        ]
        n_classes = len(self.class_names)

        self.fc1 = nn.Linear(self.bert_model.config.hidden_size, 512)
        self.out = nn.Linear(512, n_classes)

        self.scheduler = None
        self.optimizer = None
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Input data.
            attention_mask: Attention mask value.

        Returns:
            output: returns the category
        """
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        output = F.relu(self.fc1(output.pooler_output))
        output = self.drop(output)
        output = self.out(output)
        return output

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        Args:
            train_batch: Batch data
            batch_idx: Batch indices

        Returns:
            output: Training loss
        """
        input_ids = train_batch["input_ids"].to(self.device)
        attention_mask = train_batch["attention_mask"].to(self.device)
        targets = train_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.cross_entropy(output, targets)
        self.log("train_loss", loss)
        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model

        Args:
            test_batch: Batch data
            batch_idx: Batch indices

        Returns:
            output - Testing accuracy
        """
        input_ids = test_batch["input_ids"].to(self.device)
        attention_mask = test_batch["attention_mask"].to(self.device)
        targets = test_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)
        test_acc = torch.tensor(accuracy_score(y_hat.cpu(), targets.cpu()))
        self.test_outputs.append(test_acc)
        return {"test_acc": test_acc}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        Args:
            val_batch: Batch data
            batch_idx: Batch indices

        Returns:
            output: valid step loss
        """

        input_ids = val_batch["input_ids"].to(self.device)
        attention_mask = val_batch["attention_mask"].to(self.device)
        targets = val_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.cross_entropy(output, targets)
        self.val_outputs.append(loss)
        return {"val_step_loss": loss}

    def on_validation_epoch_end(self):
        """
        Computes average validation accuracy
        """
        avg_loss = torch.stack(self.val_outputs).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.val_outputs.clear()

    def on_test_epoch_end(self):
        """
        Computes average test accuracy score
        """
        print(self.test_outputs)
        avg_test_acc = torch.stack(self.test_outputs).mean()
        self.log("avg_test_acc", avg_test_acc)
        self.test_outputs.clear()

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        Returns:
            output: Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]
