"""
Parsing for the Williston 2050 Goals tables
"""

import re


def parse_goals_table(table: list[list]):
    assert table[0][0].startswith("Goals: In 2050")

    def clean(text):
        # Remove leading dots, ellipsis, and spaces
        if not text:
            return ""
        res = re.sub(r"^[.…]+\s*", "", text.strip())
        return res

    if len(table) == 7:
        assert table[1][0] == "Livable", f"Expected Livable, found {table[1][0]}"
        assert table[3][0] == "Resilient", f"Expected Resilient, found {table[3][0]}"
        assert table[5][0] == "Equitable", f"Expected Equitable, found {table[5][0]}"

        return {
            "livable": clean(table[2][0]),
            "resilient": clean(table[4][0]),
            "equitable": clean(table[6][0]),
        }

    if len(table) == 4:
        values = ["Livable", "Resilient", "Equitable"]
        result = {}
        for i, value in enumerate(values):
            pattern = re.compile(rf"^{value}\s*\n(.*)", re.DOTALL)
            match = pattern.match(table[i + 1][0])
            assert match, (
                f"Expected {value} followed by description, found {table[i + 1][0]}.\nTable: {table}"
            )
            result[value.lower()] = clean(match.group(1))

        return result

    else:
        raise Exception("Unexpected Goals Table format", table)
