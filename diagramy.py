import random
import networkx as nx
import matplotlib.pyplot as plt

def random_integer_set(lower=1, upper=10, size=5):
    """
    Generate a sorted list of unique random integers within a specified range.

    Parameters:
        lower (int): Lower bound (inclusive).
        upper (int): Upper bound (inclusive).
        size (int): Number of unique integers to generate.

    Returns:
        list[int]: Sorted list of unique random integers.

    Raises:
        ValueError: If the range is too small to accommodate the requested number of elements.
    """
    if upper - lower + 1 < size:
        raise ValueError("Range too small for the requested set size.")
    return sorted(random.sample(range(lower, upper + 1), size))


def partitions(given_set):
    """
    Generate all possible partitions of a given set.

    Parameters:
        given_set (list): The list of elements to partition.

    Returns:
        list[list[list]]: A list of partitions, where each partition is a list of blocks (which are lists themselves).
    """
    if not given_set:
        return [[]]

    first = given_set[0]
    rest_partitions = partitions(given_set[1:])
    result = []

    for smaller in rest_partitions:
        for n, subset in enumerate(smaller):
            new_partition = smaller[:n] + [[first] + subset] + smaller[n + 1:]
            result.append(new_partition)
        result.append([[first]] + smaller)

    return result

def is_noncrossing(partition):
    """
    Check if a partition is non-crossing.

    A partition is non-crossing if no four elements a < b < c < d exist such that
    a and c belong to one block, b and d to another, and the two blocks are distinct.

    Parameters:
        partition (list[list]): A partition represented as a list of blocks.

    Returns:
        bool: True if the partition is non-crossing, False otherwise.
    """
    element_to_block = {}
    for block_index, block in enumerate(partition):
        for element in block:
            element_to_block[element] = block_index

    sorted_elements = sorted(element_to_block.keys())
    n = len(sorted_elements)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    a, b, c, d = sorted_elements[i], sorted_elements[j], sorted_elements[k], sorted_elements[l]
                    if (element_to_block[a] == element_to_block[c] and
                        element_to_block[b] == element_to_block[d] and
                        element_to_block[a] != element_to_block[b]):
                        return False
    return True

def non_crossing_partitions(given_set):
    """
    Generate all non-crossing partitions of a given set.

    Parameters:
        given_set (list): The list of elements to partition.

    Returns:
        list[list[list]]: A list of non-crossing partitions.
    """
    if not given_set:
        return [[]]

    first = given_set[0]
    rest_partitions = partitions(given_set[1:])
    result = []

    for smaller in rest_partitions:
        for n, subset in enumerate(smaller):
            new_partition = smaller[:n] + [[first] + subset] + smaller[n + 1:]
            if is_noncrossing(new_partition):
                result.append(new_partition)
        result.append([[first]] + smaller)

    return result

def is_refinement(partition1, partition2):
    """
    Determine whether partition1 is a refinement of partition2.

    Parameters:
        partition1 (list[list]): First partition.
        partition2 (list[list]): Second partition.

    Returns:
        bool: True if partition1 is a refinement of partition2, False otherwise.
    """
    elem_to_block = {}
    for i, block in enumerate(partition2):
        for x in block:
            elem_to_block[x] = i
    for block in partition1:
        block_ids = {elem_to_block[x] for x in block}
        if len(block_ids) > 1:
            return False
    return True


def all_partition_pairs(partition_list):
    """
    Generate all ordered pairs (p1, p2) such that p1 is a refinement of p2.

    Parameters:
        partition_list (list[list[list]]): A list of partitions.

    Returns:
        list[tuple]: List of pairs (p1, p2) where p1 <= p2 in refinement order.
    """
    n = len(partition_list)
    result = []

    for i in range(n):
        for j in range(n):
            if i != j:
                p1, p2 = partition_list[i], partition_list[j]
                if is_refinement(p1, p2):
                    result.append((p1, p2))

    return result


def normalize(partition):
    """
    Normalize partition for use as a hashable node in a graph.

    Parameters:
        partition (list[list]): A partition.

    Returns:
        tuple: Normalized partition (tuple of sorted tuples).
    """
    return tuple(sorted(tuple(sorted(block)) for block in partition))


def build_hasse_edges(partition_list):
    """
    Build nodes and edges for a Hasse diagram (without using networkx).

    Parameters:
        partition_list (list[list[list]]): List of partitions.

    Returns:
        tuple: (nodes, edges) where nodes are normalized partitions, edges are tuples (a, b).
    """
    nodes = [normalize(p) for p in partition_list]
    edges = []
    for a, b in all_partition_pairs(partition_list):
        na, nb = normalize(a), normalize(b)
        if is_refinement(a, b):
            is_cover = True
            for c in partition_list:
                nc = normalize(c)
                if nc != na and nc != nb:
                    if is_refinement(a, c) and is_refinement(c, b):
                        is_cover = False
                        break
            if is_cover:
                edges.append((na, nb))
    return nodes, edges

def draw_hasse_lattice_manual(nodes, edges, title):
    """
    Draw a Hasse diagram using matplotlib.

    Parameters:
        nodes (list[tuple]): List of normalized partitions (nodes).
        edges (list[tuple]): List of directed edges between nodes.
        title (str): Title of the plot.

    Saves:
        A PNG image of the plotted Hasse diagram.
    """
    levels = {}
    for node in nodes:
        level = len(node)
        if level not in levels:
            levels[level] = []
        levels[level].append(node)

    sorted_levels = sorted(levels.items())

    pos = {}
    y_spacing = 2
    x_spacing = 3
    levels = len(sorted_levels)
    for i, (lvl, level_nodes) in enumerate(sorted_levels):
        y = -i * y_spacing
        count = len(level_nodes)
        for j, node in enumerate(sorted(level_nodes)):
            x = j * x_spacing - (count - 1) * x_spacing / 2
            pos[node] = (x, y)

    plt.figure(figsize=(18, 12))
    ax = plt.gca()

    for node, (x, y) in pos.items():
        plt.scatter(x, y, s=500, c='red', zorder=2, marker='s')
        label = '\n'.join(str(list(b)) for b in node)
        plt.text(x, y, label, fontsize=7, ha='center', va='center', zorder=3)

    for a, b in edges:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        plt.plot([x1, x2], [y1, y2], 'k-', zorder=1)

    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"hasse{levels}.png")
#
# for i in range(6, 7):
#     s = random_integer_set(1, i, i)
#
#     noncrossing_parts = non_crossing_partitions(s)
#
#     nodes, edges = build_hasse_edges(noncrossing_parts)
#     draw_hasse_lattice_manual(nodes, edges, f"Relacje pomiędzy nieprzecinającymi się podziałami zbioru {s}")