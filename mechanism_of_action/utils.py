import csv
from typing import Any, Dict, List


def extract_column_from_csv(filename: str,
                            column_name: str,
                            file_delimiter: str = '\t',
                            entry_delimiter: str = None) -> List[str]:
    """
    :param str filename: the CSV file from which the column will be extracted.
    :param str column_name: the name of the file which will be extracted.
    :param str file_delimiter: the delimiter used by the CSV.
    :param str entry_delimiter: the delimiter to split rows on, if any.

    :return List[str]: a list of the values in the desired column.
    """
    column = []
    with open(filename) as file_:
        reader = csv.reader(file_, delimiter=file_delimiter)

        line_count = 0
        column_index = 0

        for row in reader:
            if line_count == 0:
                column_index = row.index(column_name)
            else:
                if entry_delimiter is None:
                    column.append(row[column_index])
                else:
                    column += row[column_index].split(entry_delimiter)

            line_count += 1

    return column


def write_rows_to_csv(filename: str,
                      column_names: List[str],
                      column_data: List[Dict[str, Any]],
                      file_delimiter: str = '\t',
                      entry_delimiter: str = ', '):
    """
    """
    with open(filename, 'w+') as file_:
        writer = csv.writer(file_, delimiter='\t')
        writer.writerow(column_names)

        for i in range(len(column_data)):
            row = []
            for column in column_names:
                value = column_data[i][column]
                if isinstance(value, (list, set)):
                    row.append(entry_delimiter.join(value))
                else:
                    row.append(value)
            writer.writerow(row)
