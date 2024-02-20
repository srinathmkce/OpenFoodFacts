import pytorch_lightning as L
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from bert_dataset import OpenFoodDataset


class BertDataModule(L.LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, num_samples):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.MAX_LEN = 512
        self.encoding = None
        self.tokenizer = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset
        self.num_samples = num_samples
        self.RANDOM_SEED = 42

    def setup(self, stage=None):
        """
        Split the data into train, test, validation data

        Args:
            stage: Stage - training or testing
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

        # Load the dataset here
        #         self.dataset = pd.read_csv("en.openfoodfacts.org.products.csv", on_bad_lines='skip', sep='\t')

        self.train_dataset, self.test_dataset = train_test_split(
            self.dataset,
            test_size=0.2,
            random_state=self.RANDOM_SEED,
            stratify=self.dataset["target"],
        )
        self.val_dataset, self.test_dataset = train_test_split(
            self.test_dataset,
            test_size=0.5,
            random_state=self.RANDOM_SEED,
            stratify=self.test_dataset["target"],
        )

    def create_data_loader(self, source, count):
        """
        Generic data loader function

        Args:
            df: Input dataframe.
            tokenizer: bert tokenizer.

        Returns:
            Returns the constructed dataloader.
        """
        ds = OpenFoodDataset(
            source=source,
            tokenizer=self.tokenizer,
            max_length=self.MAX_LEN,
            num_samples=count,
            dataset=self.dataset,
        )

        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self):
        """
        Returns:
            output: Train data loader for the given input.
        """
        return self.create_data_loader(source=self.train_dataset, count=self.train_count)

    def val_dataloader(self):
        """
        Returns:
            output: Validation data loader for the given input.
        """
        return self.create_data_loader(source=self.val_dataset, count=self.val_count)

    def test_dataloader(self):
        """
        Returns:
            output: Test data loader for the given input.
        """
        return self.create_data_loader(source=self.test_dataset, count=self.test_count)
