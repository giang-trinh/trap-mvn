import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as etree
from typing import Generator, IO, List, Optional, Set

import networkx as nx

def get_actual_id(rep_id: str, rep_id_2_act_id: dict) -> str:
    if rep_id.startswith("p"):
        return rep_id_2_act_id[rep_id]
    elif rep_id.startswith("t"):
        return rep_id
    else:
        raise ValueError("Format error in place/transition")

def read_pnml(infile: IO, semantics: str):
    one_safe_PN = nx.DiGraph()
    one_safe_PN_unitary = nx.DiGraph()
    node_2_domain = {}
    rep_id_2_act_id = {}

    """
        Read the .sbml file to get the one-safe PN encoding (under general or unitary semantics) and the domain list
    """

    root = etree.parse(infile).getroot()
    if root.tag != "pnml":
        raise ValueError("Currently limited to parsing PNML files")

    for place in root.findall("./net/page/place"):
        rep_id = place.get("id")
        name = place.find("./name/text").text
        act_id = "p" + name.replace("=", "_b")
        node = name.split("=")[0]
        node_2_domain[node] = int(name.split("=")[1]) + 1

        one_safe_PN.add_node(
            act_id, kind="place"
        )

        if semantics == "unitary":
            one_safe_PN_unitary.add_node(
                act_id, kind="place"
            )

        rep_id_2_act_id[rep_id] = act_id

        #print(node + "," + act_id + "," + str(node_2_domain[node]))

    #print (",".join(str(node_2_domain[node]) for node in node_2_domain.keys()))

    for transition in root.findall("./net/page/transition"):
        one_safe_PN.add_node(transition.get("id"), kind="transition")

        if semantics == "unitary":
            one_safe_PN_unitary.add_node(transition.get("id"), kind="transition")

    for arc in root.findall("./net/page/arc"):
        #print (get_actual_id(arc.get("source"), rep_id_2_act_id) + "->" + get_actual_id(arc.get("target"), rep_id_2_act_id))
        one_safe_PN.add_edge(get_actual_id(arc.get("source"), rep_id_2_act_id), get_actual_id(arc.get("target"), rep_id_2_act_id))
    
    # TODO support unitary semantics
    if (semantics == "unitary"):
        for node, kind in one_safe_PN.nodes(data="kind"):
            if kind == "place":
                continue
            else:  # it's a transition, apply siphon (if one succ is true, one pred must be true)
                preds = list(one_safe_PN.predecessors(node))
                sucs = list(one_safe_PN.successors(node))

                node_source = str((set(preds) - set(sucs)).pop())
                node_target = str((set(sucs) - set(preds)).pop())

                sucs.remove(node_target)

                #print(str(node_source) + " --> " + str(node_target))
                value_source = get_value_from_place(node_source)
                value_target = get_value_from_place(node_target)

                new_node_source = node_source
                new_node_target = change_node_target(node_target, value_source, value_target)

                sucs.append(new_node_target)

                for pred in preds:
                    one_safe_PN_unitary.add_edge(pred, node)

                for suc in sucs:
                    one_safe_PN_unitary.add_edge(node, suc)


    if (semantics == "unitary"):
        return one_safe_PN_unitary, node_2_domain
    else:
        return one_safe_PN, node_2_domain


def get_value_from_place(place: str):
    return int(((place.split("_"))[-1])[1:])


def change_node_target(node_target: str, value_source: int, value_target: int):
    place_name = ""
    sub_strings = node_target.split("_")

    for i in range(len(sub_strings) - 1):
        place_name += sub_strings[i] + "_"

    dif = value_target - value_source
    if dif < 0:
        dif = -dif

    if dif > 1:
        print("different")

    if value_source < value_target:
        place_name += "b" + str(value_source + 1)
    else:
        place_name += "b" + str(value_source - 1)

    #print(str(value_source) + "--" + str(value_target))
    return place_name
