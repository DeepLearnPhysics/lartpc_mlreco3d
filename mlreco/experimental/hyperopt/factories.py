from . import search


def hyperopt_algorithm_dict():
	d = {
		'random_search': search.RandomSearch
	}
	return d


def construct_hyperopt_run(name):
	algs = hyperopt_algorithm_dict()
	if name not in algs:
		raise Exception("Unknown hyperopt algorithm name provided: %s" % name)
	return algs[name]