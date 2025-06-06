"""
Parsing for the 3 Facts tables
"""


def parse_facts_table(table: list[list]):
    if len(table) == 7:
        return [
            {"title": table[1][0], "text": table[2][0]},
            {"title": table[3][0], "text": table[4][0]},
            {"title": table[5][0], "text": table[6][0]},
        ]
    raise Exception(
        f"Unexpected Facts Table format of length {len(table)}",
        table,
    )
