from collections import defaultdict, Counter, namedtuple
from typing import NamedTuple, List


class StockKeepingUnit(NamedTuple): # not namedtuple?
    name: str
    description: str
    price: float
    category: str
    reviews: List[str]


d = defaultdict(int)
d["foo"] = 5
# noinspection PyArgumentList
data = {
    "foo":         [
        1,
        "Hello World!",
        100.123,
        323.232,
        432324.0,
        {5, 6, 7, (1, 2, 3, 4), 8},
        ],
    "bar":         frozenset({1, 2, 3}),
    "defaultdict": defaultdict(
            list, {"crumble": ["apple", "rhubarb", "butter", "sugar", "flour"]}
            ),
    "counter":     Counter(
            [
                "apple",
                "orange",
                "pear",
                "kumquat",
                "kumquat",
                "durian" * 100,
                ]
            ),
    "atomic":      (False, True, None),
    "namedtuple":  StockKeepingUnit(
            "Sparkling British Spring Water",
            "Carbonated spring water",
            0.9,
            "water",
            ["its amazing!", "its terrible!"],
            ),
    }
data["foo"].append(data['namedtuple'])  # type: ignore[attr-defined]