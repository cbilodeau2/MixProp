"""Selects the ZINC tranches which have the highest proportion of antibiotics."""
from collections import defaultdict
import csv
import os
from typing import Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
from tap import Tap  # pip install typed-argument-parser https://github.com/swansonk14/typed-argument-parser


ZINC_LETTERS = 'ABCDEFGHIJK'
ZINC_TRANCHES = [f'{l1}{l2}' for l1 in ZINC_LETTERS for l2 in ZINC_LETTERS]
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
    data_path: str  # Path to CSV file containing molecules with tranches labelled by compute_zinc_tranches.py.
    save_path: str  # Path to CSV file where ratios will be saved.
    plot: bool = False  # Whether to plot tranche sizes and scores.


def plot_tranche_sizes(tranche_sizes: Dict[str, Union[int, float]], title: str = None) -> None:
    """Plots ZINC tranche sizes."""
    size_grid = [
        [tranche_sizes[f'{l2}{l1}'] for l2 in ZINC_LETTERS]
        for l1 in ZINC_LETTERS
    ]
    plt.imshow(size_grid, cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    plt.xticks(ticks=range(len(ZINC_LETTERS)), labels=list(ZINC_LETTERS))
    plt.yticks(ticks=range(len(ZINC_LETTERS)), labels=list(ZINC_LETTERS))
    plt.xlabel('Molecular Weight')
    plt.ylabel('LogP')
    plt.title('Tranche sizes' if title is None else title)
    plt.show()


def select_zinc_trances(args: Args) -> None:
    """Selects the ZINC tranches which have the highest proportion of antibiotics."""
    # Load data
    antibiotics = pd.read_csv(args.antibiotics_path)
    antibiotic_tranche_sizes = defaultdict(int, antibiotics['tranche'].value_counts())

    data = pd.read_csv(args.data_path)
    data_tranche_sizes = defaultdict(int, data['tranche'].value_counts())

    # Compute ratios of antibiotics to data or ZINC
    antibiotic_data_ratio = defaultdict(float, {
        tranche: antibiotic_tranche_sizes[tranche] / data_tranche_sizes[tranche] if data_tranche_sizes[tranche] > 0 else 0
        for tranche in ZINC_TRANCHES
    })
    antibiotic_zinc_ratio = defaultdict(float, {
        tranche: antibiotic_tranche_sizes[tranche] / ZINC_TRANCHE_SIZES[tranche]
        for tranche in ZINC_TRANCHES
    })

    # Plot tranche sizes/ratios
    if args.plot:
        plot_tranche_sizes(antibiotic_tranche_sizes, title='Antibiotic tranche sizes')
        plot_tranche_sizes(data_tranche_sizes, title='Data tranche sizes')
        plot_tranche_sizes(ZINC_TRANCHE_SIZES, title='ZINC tranche sizes')
        plot_tranche_sizes(antibiotic_data_ratio, title='Antibiotic / data ratio')
        plot_tranche_sizes(antibiotic_zinc_ratio, title='Antibiotic / ZINC ratio')

    # Create save directory
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)

    # Save ratios
    with open(args.save_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Tranche',
            'Antibiotic / data ratio',
            'Antibiotic / ZINC ratio',
            '# Antibiotics',
            '# Data',
            '# ZINC'
        ])

        for tranche in ZINC_TRANCHES:
            writer.writerow([
                tranche,
                antibiotic_data_ratio[tranche],
                antibiotic_zinc_ratio[tranche],
                antibiotic_tranche_sizes[tranche],
                data_tranche_sizes[tranche],
                ZINC_TRANCHE_SIZES[tranche]
            ])


if __name__ == '__main__':
    select_zinc_trances(Args().parse_args())
