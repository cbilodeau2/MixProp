"""Selects the ZINC tranches which have the highest proportion of antibiotics."""
from typing_extensions import Literal

import pandas as pd
from tap import Tap  # pip install typed-argument-parser https://github.com/swansonk14/typed-argument-parser

ZINC_LETTERS = 'ABCDEFGHIJK'
ZINC_TRANCHE_SIZES = [
    [29293, 204598, 784279, 1125069, 2321356, 854208, 300607, 128558, 99872, 86323, 5615],
    [142690, 1067035, 3992760, 5372590, 10975901, 3784188, 1767726, 775279, 606137, 558305, 3798],
    [376413, 3284847, 13196175, 17023840, 34876129, 12665806, 7279946, 3517752, 2892839, 2688060, 7987],
    [497750, 5391816, 25622912, 32914848, 67733100, 28989280, 19267814, 10563987, 9000177, 8721010, 20894],
    [189326, 2643678, 14831118, 19486349, 40600593, 20281809, 15126147, 9325848, 8120159, 7879918, 21325],
    [108266, 2075334, 13281388, 18060096, 37030641, 22002857, 17838728, 12045788, 10696251, 10674949, 33982],
    [48705, 1336320, 10135959, 14349999, 29671752, 21055698, 18737428, 13954286, 12511433, 12736846, 54896],
    [15100, 613109, 6131454, 8128568, 12531547, 15472307, 16892846, 14129429, 12864529, 13378049, 82058],
    [1993, 170043, 2873064, 4632339, 7889352, 10959424, 12773295, 12356636, 11562431, 12208182, 113230],
    [94, 21765, 852691, 1919530, 3985092, 6416455, 8397678, 9021197, 8818548, 9321913, 139087],
    [28, 884, 44519, 175953, 549357, 1226923, 2066211, 2628127, 3062143, 3771646, 735850]
]
ZINC_TRANCHE_SIZES = {
    f'{l2}{l1}': ZINC_TRANCHE_SIZES[i][j]
    for j, l2 in enumerate(ZINC_LETTERS)
    for i, l1 in enumerate(ZINC_LETTERS)
}


class Args(Tap):
    antibiotics_path: str  # Path to CSV file containing antibiotics with tranches labelled by compute_zinc_tranches.py.
    data_path: str = None  # Path to CSV file containing molecules with tranches labelled by compute_zinc_tranches.py.
    selection_type: Literal['zinc', 'data', 'number'] = 'zinc'  # Type of tranche selection to perform.
    """
    'zinc' computes the ratio of antibiotics per tranche to ZINC molecules per tranche.
    'data' computes the ratio of antibiotics per tranche to data molecules per tranche, with data from data_path.
    'number' computes the raw number of antibiotics per tranche (i.e., no normalization).
    """
    top_k: int = None  # The number of tranches to select.


def select_zinc_trances(args: Args) -> None:
    """Selects the ZINC tranches which have the highest proportion of antibiotics."""
    # Checks
    if (args.selection_type == 'data') != (args.data_path is not None):
        raise ValueError('data selection type can be used if and only if a data path is provided.')

    # Load data
    antibiotics = pd.read_csv(args.antibiotics_path)
    antibiotic_tranches_sizes = antibiotics['tranche'].value_counts()

    # Sort tranches depending on selection type
    if args.selection_type == 'zinc':
        tranche_scores = {
            tranche: antibiotic_tranches_sizes[tranche] / ZINC_TRANCHE_SIZES[tranche]
            for tranche in antibiotic_tranches_sizes.keys()
        }
    elif args.selection_type == 'data':
        data = pd.read_csv(args.data_path)
        data_tranche_sizes = data['tranche'].value_counts()

        tranche_scores = {
            tranche: antibiotic_tranches_sizes[tranche] / data_tranche_sizes[tranche]
            for tranche in antibiotic_tranches_sizes.keys()
        }
    elif args.selection_type == 'number':
        tranche_scores = antibiotic_tranches_sizes
    else:
        raise ValueError(f'Selection type "{args.selection_type}" not supported.')

    # Select top tranches
    tranches = sorted(antibiotic_tranches_sizes.keys(),
                      key=lambda tranche: tranche_scores[tranche],
                      reverse=True)

    # Print top tranches
    for tranche in tranches[:args.top_k]:
        print(f'{tranche}\t{tranche_scores[tranche]:.6f}')


if __name__ == '__main__':
    select_zinc_trances(Args().parse_args())
