import logging
import time
import numpy as np
import os
import plotly.express as px
import sklearn.datasets
import sklearn.metrics.cluster

import clustering_algorithms
import clusters_data
import data
import draw

import cProfile
import re


logger = logging.getLogger('k_mxt_w')


def main():
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('k_mxt_w.log')
    formatter = logging.Formatter('%(filename)s func:%(funcName)s [LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]' +
                                  '%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info('Program started')
    try:
#         # filename = '../data/homegate_month_living_dropna.csv'
#         # data_property = data.DataPropertyImportSpace(filename=filename)
#         #
#         # features_list = ['price']
#         # x, y, features = data_property.get_data(features_list=features_list)
        for k in [9]:
            for eps in [0.4]:
                coord, labels = sklearn.datasets.make_moons(n_samples=50, noise=0.05, random_state=0)
                clusters = clusters_data.ClustersDataSpace2d(x_init=coord[:, 0], y_init=coord[:, 1], metrics='euclidean')
                alg = clustering_algorithms.K_MXT_gauss(k=k, eps=eps, clusters_data=clusters)
                start_time = time.time()
                alg()
                end_time = time.time()
                print(clusters.cluster_numbers)
                print(f'k-{k}, eps-{eps}, time-{end_time - start_time}')
                print(f'ARI-{sklearn.metrics.cluster.adjusted_rand_score(clusters.cluster_numbers, labels)}')
        #         data.DataSave.arrays_to_csv(source_filename = filename,
        #                                     new_filename=f'./results/{os.path.basename(filename)}_k_{k}_eps_{eps}.csv',
        #                                     cluster_numbers=clusters.cluster_numbers)
        # draw.Draw.draw_points(filename=filename, lat='latitude', lon='longitude', hover_name='price',
        #                       hover_data=['rooms', 'url', 'property_type'],
        #                       color='price',
        #                       color_continuous_scale=px.colors.cyclical.HSV)
        # draw.Draw.draw_points(filename=filename, lat='latitude', lon='longitude',
        #                       hover_name='rent_per_room',
        #                       hover_data=['price', 'rooms'],
        #                       color='rent_per_room',
        #                       )

    except BaseException as e:
        logger.error(e, exc_info=True)
    logger.info('Done!')


if __name__ == '__main__':
    main()

