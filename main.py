from itertools import product
from Dataset import Dataset
from Test import Test
import pandas as pd
import copy


def main():
    '''
    For each dataset, this function will first retrieve it, then calculate its features and finally run a series of tests on it.
    The result is saved in a csv file (one per dataset) along with the dataset information
    '''

    datasets = ['cifar100', 'cifar10', 'mnist', 'fashion_mnist',
                'coil20', 'intel_img_class', 'shipsnet', 'usps', 'umist']

    print('#################### START ####################\n')

    # start the tests
    for dataset_name in datasets:
        # initialize results dataframe
        results = pd.DataFrame()

        # retrieve dataset
        print('\nRetrieving {}...'.format(dataset_name))
        dataset = getattr(Dataset, dataset_name)()
        print('Done.\n')

        print('Calculating dataset specific features...')
        dataset_features = Dataset.get_dataset_features(
            dataset.X_train, dataset.y_train)
        print('Done.\n')

        # create every combination of the neighbor and component values to be tested
        components_tests = list(range(1, dataset_features['intr_dim']*2 + 1))
        neighbors_tests = [2, 5, 10, 20, 50]

        # the previous two variables are combined to create every possible value combination between their elements
        tests = [dict(zip(('n_components', 'n_neighbors'), (i, j)))
                 for i, j in product(*[components_tests, neighbors_tests])]

        # test knn on the dataset and save the results to the dataframe
        print('\tRunning tests on {dataset}'.format(dataset=dataset.name))

        for test_params in tests:
            # initialize test object and deepcopy the dataset to not modify the original dataset
            test = Test(copy.deepcopy(dataset), test_params)

            result = test.run()
            # add the dataset features to the result dictionary
            result.update(dataset_features)

            # save results to the dataframe
            results = results.append(result, ignore_index=True)

        print('Tests on {} finished.'.format(dataset.name))

        print('Saving results to a file...')
        results.to_csv('results/' + dataset.name + '.csv',
                       index=False, float_format='%.3f')
        print('Done.\n')

    print('#################### END ####################')


if __name__ == '__main__':
    main()
