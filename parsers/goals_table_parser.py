"""
Parsing for the Williston 2050 Goals tables
"""


def parse_goals_table(table: list[list]):
    assert table[0][0].startswith("Goals: In 2050")

    if len(table) == 7:
        assert table[1][0] == "Livable", f"Expected Livable, found {table[1][0]}"
        assert table[3][0] == "Resilient", f"Expected Resilient, found {table[3][0]}"
        assert table[5][0] == "Equitable", f"Expected Equitable, found {table[5][0]}"

        return (
            {
                "livable": table[2][0].strip(),
                "resilient": table[4][0].strip(),
                "equitable": table[6][0].strip(),
            },
        )
    else:
        raise Exception("Unexpected Goals Table format", table)
