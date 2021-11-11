from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import time
import umap


class Test:
    ''' 
    This class is used to run a test with defined hyperparameter values and dataset as input. The test takes care of initializing a new UMAP object every time which is used to embed the dataset. The embedded dataset is then scaled and fed to KNN. The predict method return the accuracy of the model.
    '''

    def __init__(self, dataset, params=None):
        self.dataset = dataset
        self.params = params
        self.result = {}

    def run(self):
        print('\t\tStarting test with parameters: n_neighbors = {}, n_components = {}'.format(
            self.params['n_neighbors'], self.params['n_components']))
        start_time = time.process_time()

        # initialize UMAP object
        reducer = umap.UMAP(n_neighbors=self.params['n_neighbors'],
                            n_components=self.params['n_components'], min_dist=0.0)

        # create embedded dataset
        reducer.fit(self.dataset.X_train)
        self.dataset.X_train = reducer.transform(self.dataset.X_train)
        self.dataset.X_test = reducer.transform(self.dataset.X_test)
        print('\t\tEmbedding finished.')

        # scale the embedded dataset to values between 0 and 1
        scaler = MinMaxScaler()
        self.dataset.X_train = scaler.fit_transform(self.dataset.X_train)
        self.dataset.X_test = scaler.transform(self.dataset.X_test)

        # predict the labels of the test set
        y_pred = self.predict()

        execution_time = time.process_time() - start_time
        print('\t\tTime taken: {} seconds'.format(
            execution_time))

        accuracy = accuracy_score(self.dataset.y_test, y_pred)
        print('\t\tAccuracy on {} with UMAP is: {}\n'.format(
            self.dataset.name, accuracy))
        self.result = {'dataset': self.dataset.name, 'accuracy': accuracy, 'time': int(
            execution_time), 'n_neighbors': self.params['n_neighbors'], 'n_components': self.params['n_components']}
        return self.result

    def predict(self):
        knn = KNeighborsClassifier()
        knn.fit(self.dataset.X_train, self.dataset.y_train)
        y_pred = knn.predict(self.dataset.X_test)
        return y_pred
