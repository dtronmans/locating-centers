import json


class Config:

    def __init__(self, json_path):
        with open(json_path, 'r') as file:
            config = json.load(file)

        self.dataset_path = self.get_nested(config, ['train', 'dataset_path'])
        self.num_epochs = self.get_nested(config, ['train', 'num_epochs'])
        self.learning_rate = self.get_nested(config, ['train', 'learning_rate'])
        self.batch_size = self.get_nested(config, ['train', 'batch_size'])
        self.val_split = self.get_nested(config, ['train', 'val_split'])
        self.num_channels = self.get_nested(config, ['train', 'num_channels'])

        self.predict_image = self.get_nested(config, ['infer', 'image_path'])

    def __repr__(self):
        return f"Config({self.__dict__})"

    def get_nested(self, dictionary, keys, default=None):
        for key in keys:
            if isinstance(dictionary, dict):
                dictionary = dictionary.get(key)
            else:
                return default
        return dictionary
