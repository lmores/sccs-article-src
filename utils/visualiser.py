import sys

from matplotlib import pyplot, figure
from upsetplot import from_contents, UpSet

from instance import read_instance


SIZE = 50


def main():
    instance_id = sys.argv[1]
    data = read_instance(f'beasley/{instance_id}-3.txt', 1)
    d = from_contents(dict(enumerate(data.subsets)))
    us = UpSet(d, sort_categories_by=None)
    fig = figure.Figure()
    us.plot(fig)
    fig.savefig(f'{instance_id}-3.png')
    pyplot.show()


if __name__ == '__main__':
    main()
