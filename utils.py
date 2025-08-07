import random
import networkx as nx
import matplotlib.pyplot as plt
import time

def random_integer_set(lower=1, upper=10, size=5):
    """
    Returns set of *size* elements between *lower* and *upper* inclusive.
    :param lower: lower bound
    :param upper: upper bound
    :param size: cardinality of set.
    """
    if upper - lower + 1 < size:
        raise ValueError("Range too small for the requested set size.")
    return sorted(random.sample(range(lower, upper + 1), size))

def is_partition(candidate_for_partition):
    """
    Returns True if *given_set* is a proper partition (by the definition)
    :param given_set:
    """
    used_integers = []
    for block in candidate_for_partition:
        for element in block:
            if element not in used_integers:
                used_integers.append(element)
            else:
                return False

# def is_partition_of_some_set(partition, given_set):
#     if is_partition(partition):
#
#     else:
#         return False


def partitions(given_set):
    """
    Returns a list of all partitions of the given set.

    :param given_set: list of elements to partition
    :return: list of partitions, where each partition is a list of blocks (lists)
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


def random_partition(s):
    """Randomly partitions a list `s` into a partition (list of non-empty, disjoint subsets)."""
    return partitions(s)[random.randint(0, len(s) - 1)]



def is_noncrossing(partition):
    """
    Returns True if partition is non-crossing.
    :param partition: partition of
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
    Returns a list of all non-crossing partitions of the given set.

    :param given_set: list of elements to partition
    :return: list of partitions, where each partition is a list of blocks (lists)
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
    Returns True if partition1 <= partition2, where "<=" is a refinement order
    :param partition1: first partition
    :param partition2: second partition
    :return:
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
    Returns all ordered pairs (p1, p2) from the list where p1 is a refinement of p2.

    :param partition_list: list of partitions (each a list of blocks)
    :return: list of tuples (p1, p2) such that p1 <= p2 in refinement order
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
    return tuple(sorted(tuple(sorted(block)) for block in partition))

def build_hasse_graph(partition_list):
    G = nx.DiGraph()

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
                G.add_edge(na, nb)
    return G

def draw_hasse_diagram(G, title="Hasse Diagram"):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=800, node_color='lightyellow', edge_color='gray', arrows=True)
    labels = {n: str([list(b) for b in n]) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.show()


def build_hasse_edges(partition_list):
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


def benchmark_drawing_times(max_n=6):
    draw_times = []
    sizes = []

    for n in range(1, max_n + 1):
        s = list(range(1, n + 1))
        print(f"Generating for set: {s}")

        all_parts = list(partitions(s))
        nc_parts = [p for p in all_parts if is_noncrossing(p)]

        nodes, edges = build_hasse_edges(nc_parts)

        print(f"Partitions: {len(all_parts)} total, {len(nc_parts)} non-crossing")

        start = time.perf_counter()
        draw_hasse_lattice_manual(nodes, edges, f"Non-Crossing Lattice for n={n}")
        end = time.perf_counter()
        elapsed = end - start
        draw_times.append(elapsed)
        sizes.append(n)

        print(f"Draw time for n={n}: {elapsed:.4f} seconds")

    return sizes, draw_times

def plot_drawing_benchmark(sizes, draw_times, filename="drawing_times", size=0):
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, draw_times, marker='o', color='darkred')
    plt.xlabel("n (size of set [1..n])")
    plt.ylabel("Time to draw lattice (seconds)")
    plt.title("Time to Draw Non-Crossing Partition Lattice")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename}{size}.png")
    plt.show()

for i in range(6, 7):
    s = random_integer_set(1, i, i)

    noncrossing_parts = non_crossing_partitions(s)

    nodes, edges = build_hasse_edges(noncrossing_parts)
    draw_hasse_lattice_manual(nodes, edges, f"Relacje pomiędzy nieprzecinającymi się podziałami zbioru {s}")