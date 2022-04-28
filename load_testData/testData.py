import numpy as np


class MlpData:
    def __init__(self, train_in_path, train_out_path):
        self.train_in_path = train_in_path
        self.train_out_path = train_out_path

    def InData(self):
        tempList = []
        with open(self.train_in_path, "r") as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                line = line.strip()
                inData = line.split("   ")
                tempList.append(inData)
        return np.array(tempList, dtype=np.float32)

    def outData(self):
        tempList = []
        with open(self.train_out_path, "r") as f:
            lines = f.readlines()

            for index, line in enumerate(lines):
                line = line.strip()
                inData = line.split("   ")
                tempList.append(inData)
        return np.array(tempList, dtype=np.float32)
