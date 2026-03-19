class PIJEPAWrapper:
    def __init__(self, config):
        from models.pi_jepa import PIJEPA
        self.model = PIJEPA(config)

    def train(self, train_loader):
        from training import Trainer
        trainer = Trainer(self.model)
        trainer.train(train_loader)

    def predict(self, x):
        return self.model(x)