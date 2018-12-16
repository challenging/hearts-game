import pickle

from card import bitmask_to_str

from tree import TreeNode
from treelib import Node, Tree


node = None
current_idx = 0

def walk(data):
    global node, current_idx

    if node is None:
        node = Tree()
        node.create_node("root", current_idx)

    if current_idx >= 4096:
       return

    root_idx = current_idx
    for (suit, rank), child in data._children.items():
        current_idx += 1
        card = bitmask_to_str(suit, rank)

        node.create_node(card, current_idx, parent=root_idx)

        walk(child)


def get_tree(tree):
    global node

    walk(tree)

    return node


if __name__ == "__main__":
    data = None
    with open("memory_tree.bak/model/memory_mcts.64.pkl", "rb") as in_file:
        data = pickle.load(in_file)

    tree = get_tree(data)
    tree.show()

    print("node={}, depth={}".format(tree.size(), tree.depth()))
