import os
from operator import itemgetter

from sklearn.externals import joblib

try:
    BASE_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except:
    BASE_DIR = os.getcwd()

MODEL_DIR = os.path.join(BASE_DIR, 'data/models')


def load_models():
    results = {}
    for fname in os.listdir(MODEL_DIR):
        model_name = fname.split('.')[0]
        fpath = os.path.join(MODEL_DIR, fname)
        results[model_name] = joblib.load(fpath)
    return results


def save_models(results):
    for name, res in results.items():
        print('%s.pkl' % os.path.join(MODEL_DIR, name))
        joblib.dump(res, '%s.pkl' % os.path.join(MODEL_DIR, name))


def get_best(results):
    return sorted(results.values(), key=itemgetter('score'), reverse=True)[0]


def sort_results(results):
    return sorted(results.values(), key=itemgetter('score'), reverse=True)


def replace_if_better(res, results):
    if res['name'] not in results:
        results[res['name']] = res
        return True

    elif res['score'] > results[res['name']]['score']:
        print('!! IMPROVEMENT !!')
        results[res['name']] = res
        return True

    return False


def latex_table(data, pos='', table_spec=''):
    table = ''
    for row in data:
        if not table:
            if not table_spec:
                table_spec = 'c' * len(row)
            table_spec = '{' + table_spec + '}'
            if not pos:
                pos=''
            table = '    \\begin{{tabular}}{pos}{table_spec}\n'.format(
                pos=pos,
                table_spec=table_spec,
            )
        table += '        ' + ' & '.join(map(str, row)) + ' \\\\\n'
    table += '    \\end{tabular}\n'
    return table