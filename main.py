import logging
import sklearn.datasets
import sklearn.metrics.cluster

import clustering_algorithms
import clusters_data


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
        for k in [9]:
            for eps in [0.4]:
                coord, labels = sklearn.datasets.make_moons(n_samples=50, noise=0.05, random_state=0)
                clusters = clusters_data.ClustersDataSpace2d(x_init=coord[:, 0], y_init=coord[:, 1], metrics='euclidean')
                alg = clustering_algorithms.K_MXT_gauss(k=k, eps=eps, clusters_data=clusters)
                alg()
                print(clusters.cluster_numbers)
    except BaseException as e:
        logger.error(e, exc_info=True)
    logger.info('Done!')


if __name__ == '__main__':
    main()

