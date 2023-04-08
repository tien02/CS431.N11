import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.optim import Adam
from termcolor import colored

config = {
    "BATCH_SIZE": 128,
    "LEARNING_RATE":1e-4,
    "EPOCHS":3,
    "CKPT_PATH": "./checkpoints",
    "TENSORBORAD_LOGGER":{
        "NAME": "myCNN",
        "VERSION": 0
    }
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class MyCNN(nn.Module):
    def __init__(self, use_act=True, use_pool=True, use_fc=True):
        super(MyCNN, self).__init__()
        self.use_fc = use_fc
        # Conv 1
        conv1 = []
        conv1.append(nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0))
        conv1.append(nn.BatchNorm2d(6))
        if use_act:
            conv1.append(nn.ReLU())
        if use_pool:
            conv1.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv1 = nn.Sequential(*conv1)
        
        # Conv 2
        conv2 = []
        conv2.append(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0))
        conv2.append(nn.BatchNorm2d(16))
        if use_act:
            conv2.append(nn.ReLU())
        if use_pool:
            conv2.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(*conv2)

        # Conv 3
        conv3 = []
        conv3.append(nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0))
        conv3.append(nn.BatchNorm2d(16))
        if use_act:
            conv3.append(nn.ReLU())
        if use_pool:
            conv3.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv3 = nn.Sequential(*conv3)
        
        # Classifier
        classifier = []
        if use_fc:
            if use_pool:
                classifier.append(nn.Linear(256, 84))
                if use_act:
                    classifier.append(nn.ReLU())
                classifier.append(nn.Linear(84, 10))
            else:
                classifier.append(nn.Linear(6400, 2100))
                if use_act:
                    classifier.append(nn.ReLU())
                classifier.append(nn.Linear(2100, 10))
        else:
            if use_pool:
                classifier.append(nn.Conv2d(16, 21, kernel_size=3, stride=1, padding=0))
                if use_act:
                    classifier.append(nn.ReLU())
                classifier.append(nn.Conv2d(21, 10, kernel_size=2, stride=1, padding=0))
            else:
                classifier.append(nn.Conv2d(16, 21, kernel_size=11, stride=1, padding=0))
                if use_act:
                    classifier.append(nn.ReLU())
                classifier.append(nn.Conv2d(21, 10, kernel_size=10, stride=1, padding=0))

        self.classifer = nn.Sequential(*classifier)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_fc:
            out = out.reshape(out.size(0), -1)
        out = self.classifer(out)
        if self.use_fc:
            return out
        else:
            return torch.squeeze(torch.squeeze(out, dim=3), dim=2)

class MyModel(pl.LightningModule):
    def __init__(self, model):
        super(MyModel, self).__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = Accuracy(num_classes=10,average="macro")
        self.precision_fn = Precision(num_classes=10, average="macro")
        self.recall_fn = Recall(num_classes=10, average="macro")
        self.f1_score_fn = F1Score(num_classes=10, average="macro")

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch):
        images, labels = batch
        logits = self.model(images)
        predict_prob = nn.functional.softmax(logits, dim=1)
        loss = self.loss_fn(predict_prob, labels)

        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.model(images)
        predict_prob = nn.functional.softmax(logits, dim=1)
        loss = self.loss_fn(predict_prob, labels)
        acc = self.accuracy_fn(predict_prob, labels)
        pre = self.precision_fn(predict_prob, labels)
        re = self.recall_fn(predict_prob, labels)
        f1 = self.f1_score_fn(predict_prob, labels)
        
        self.log_dict({
            "test_loss": loss,
            "test_acc": acc,
            "test_precision": pre,
            "test_recall": re,
            "test_f1": f1
        })

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=config["LEARNING_RATE"], eps=1e-6, weight_decay=0.01)
        return optimizer

if __name__ == "__main__":
    seed_everything(42)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=trans)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=trans)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=3)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=3)

    print(colored("Training with full options", color="blue"))
    full_options_model = MyCNN()
    system = MyModel(full_options_model)
    trainer = Trainer(accelerator='gpu', gradient_clip_val=1.0,max_epochs=config["EPOCHS"])
    trainer.fit(model=system, train_dataloaders=train_dataloader)
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    print(colored("Training without acivation function", color="blue"))
    without_activation_function_model = MyCNN(use_act=False)
    system = MyModel(without_activation_function_model)
    trainer = Trainer(accelerator='gpu', gradient_clip_val=1.0,max_epochs=config["EPOCHS"])
    trainer.fit(model=system, train_dataloaders=train_dataloader)
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    print(colored("Training without pooling layer", color="blue"))
    without_pooling_layer_model = MyCNN(use_pool=False)
    system = MyModel(without_pooling_layer_model)
    trainer = Trainer(accelerator='gpu', gradient_clip_val=1.0,max_epochs=config["EPOCHS"])
    trainer.fit(model=system, train_dataloaders=train_dataloader)
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    print(colored("Training without fully connected layer", color="blue"))
    without_fc_layer_model = MyCNN(use_fc=False)
    system = MyModel(without_fc_layer_model)
    trainer = Trainer(accelerator='gpu', gradient_clip_val=1.0,max_epochs=config["EPOCHS"])
    trainer.fit(model=system, train_dataloaders=train_dataloader)
    trainer.test(ckpt_path="best", dataloaders=test_dataloader)
