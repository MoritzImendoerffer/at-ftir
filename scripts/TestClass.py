    import numpy as np
    from sklearn.preprocessing import StandardScaler

    class TestClass:

        def __init__(self, data):
            self.data = data
            self._preprocessed = data


        def preprocessing(self):

            def gradient(self):
                self._preprocessed = np.gradient(self._preprocessed, 2)[1]

            def normalize(self):
                self._preprocessed = StandardScaler().fit_transform(self._preprocessed)

        def cluster_analysis(self):

            def pca(self):
                pass