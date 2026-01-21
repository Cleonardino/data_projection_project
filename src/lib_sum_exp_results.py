#
from typing import Any, Optional

import os
import json


#
NFS: list[str] = [
    "config.yaml",
    "training_history.json",
    "metrics.json",
]


def is_ok(path: str) -> bool:
    #
    fls: list[str] = os.listdir( path )
    #
    for nf in NFS:
        #
        present: bool = False
        #
        for f in fls:
            if f.startswith(nf):
                present = True
                break
        #
        if not present:
            return False
    return True


def process_folder(path: str) -> dict[str, Any]:
    #
    with open(os.path.join(path, "training_history.json")) as f:
        h: dict[str, Any] = json.load( f )

    with open(os.path.join(path, "metrics.json")) as f:
        m: dict[str, Any] = json.load( f )

    #
    return {
        "metrics": m,
        "history": h
    }



def load_experiments(exp_dir: str = "experiments/") -> dict[str, dict[str, dict[str, Any]]]:
    #
    res: dict[str, dict[str, dict[str, Any]]] = {}
    #
    for exp_folder in os.listdir( exp_dir ):
        #
        nd: str = os.path.join( exp_dir, exp_folder )
        #
        if not is_ok(nd):
            continue
        #
        res[exp_folder] = process_folder( nd )

    #
    return res




# To test
if __name__ == "__main__":
    a = load_experiments()
    print(a.keys())
    print(len(a))
