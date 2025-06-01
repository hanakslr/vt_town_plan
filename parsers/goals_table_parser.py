"""
Parsing for the Williston 2050 Goals tables
"""

import re


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
    if len(table) == 4:
        values = ["Livable", "Resilient", "Equitable"]
        result = {}
        for i, value in enumerate(values):
            pattern = re.compile(rf"^{value}\s*\n(.*)", re.DOTALL)
            match = pattern.match(table[i + 1][0])
            assert match, (
                f"Expected {value} followed by description, found {table[i + 1][0]}.\nTable: {table}"
            )
            result[value.lower()] = match.group(1).strip()

        return result

    else:
        raise Exception("Unexpected Goals Table format", table)
