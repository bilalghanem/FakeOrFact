import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from data.data import data
from model import Model
from sklearn.pipeline import Pipeline
from features.building_features import manual_features
from sklearn.preprocessing import StandardScaler

tqdm.pandas()

if __name__ == '__main__':
    """Loading data"""
    loaded_data = data()
    loaded_data.load_data()

    """Run model"""
    model = Model()
    model.run_model(loaded_data)


    """Run Deep text only"""
    # from models.CNN import CNN
    # deep = CNN()
    # deep.prepare_input(loaded_data.train, loaded_data.dev, loaded_data.test)
    # deep.Network()
    # deep.run_model(reTrain=True)


    """Run Deep text + features"""
    # model = Pipeline([
    #     ('main_pip', Pipeline([
    #         ('manual_features', manual_features(path='./features', n_jobs=-1)),
    #         ('Normalization', StandardScaler()),
    #     ])),
    # ])
    # from models.DL_branches import DL_branches
    # x_train = model.fit_transform(loaded_data.train, loaded_data.train['label'])
    # x_dev = model.transform(loaded_data.dev)
    # x_test = model.transform(loaded_data.test)
    # deep = DL_branches()
    # deep.prepare_input_branches(loaded_data, x_train, x_dev, x_test)
    # deep.Network_branches_CNN()
    # deep.run_model(reTrain=True)