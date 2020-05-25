from abc import ABC, abstractmethod


class DataLoader(ABC):
    conf_filename = None
    project_conf_filename = None

    @abstractmethod
    def make_data(self, val_pct=None):
        pass

    @abstractmethod
    def save_Xy(self, train_X, train_y, test_X, test_y):
        pass

    def read_XY(self):
        print("Loading train_X, train_y, test_X, test_y...")
        train_X = self.read_train_X()
        train_y = self.read_train_y()
        test_X = self.read_train_X()
        test_y = self.read_train_y()
        print("Load successful")
        return train_X, test_X, train_y, test_y

    @abstractmethod
    def read_train_X(self):
        pass

    @abstractmethod
    def read_train_y(self):
        pass

    @abstractmethod
    def read_test_X(self):
        pass

    @abstractmethod
    def read_test_y(self):
        pass

    @abstractmethod
    def get_input_size(self):
        pass
