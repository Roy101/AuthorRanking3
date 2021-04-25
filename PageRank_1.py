from networkx.drawing.tests.test_pylab import plt


def pagerank(G, alpha, personalization=None,
             max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
             dangling=None):

    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    # Create a copy in (right) stochastic form
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()

    # Choose fixed starting vector if not given
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        # Normalized nstart vector
        s = float(sum(nstart.values()))
        x = dict((k, v / s) for k, v in nstart.items())

    if personalization is None:

        # Assign uniform personalization vector if not given
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise nx.NetworkXError('Personalization dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(personalization.values()))
        p = dict((k, v / s) for k, v in personalization.items())

    if dangling is None:

        # Use personalization vector if dangling vector not specified
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise nx.NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        s = float(sum(dangling.values()))
        dangling_weights = dict((k, v / s) for k, v in dangling.items())
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:

            # this matrix multiply looks odd because it is
            # doing a left multiply x^T=xlast^T*W
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

        # check convergence, l1 norm
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
    raise nx.NetworkXError('pagerank: power iteration failed to converge '
                        'in %d iterations.' % max_iter)
import networkx as nx
s = ((2,1),(3,1),(3,2),(3,4),(3,5),(4,1),(5,2),(5,6),(6,4),(7,2),(7,3),(7,4),(7,5),(7,6))
q=((2,1),(2,3),(1,6),(1,7),(3,4),(3,7),(3,8),(4,7),(4,8),(5,4),(5,6),(5,7),(5,8),(1,9),(1,10))
r=((2,1),(3,1),(4,1),(3,2),(4,2),(7,3),(8,3),(9,3),(12,3),(7,4),(10,4),(11,4),(12,4),(2,5),(7,5),(8,5),(9,5),(2,6),(5,6),(10,6),(11,6),(12,6),(8,7),(9,7),(9,8),(10,11),(10,12),(11,12))
t=((2,1),(3,1),(4,1),(5,1),(6,1),(3,2),(4,3),(5,3),(6,3),(7,3),(6,4),(7,4),(8,4),(9,4),(1,10),(2,10),(5,6),(8,6),(9,6))
u=((2,1),(3,1),(4,1),(10,1),(3,2),(2,10),(3,10),(6,10),(4,3),(5,3),(6,3),(4,5),(6,5),(6,4),(7,4),(8,4),(7,5),(8,5),(8,7))
G = nx.DiGraph()
G.add_edges_from(u)

"""plt.figure(figsize =(9, 9))
plt.savefig("pagerank.png")
nx.draw(G, node_color ='green')
plt.show()"""

pr=nx.pagerank(G,0.85)
print(sorted(pr.items(), key = lambda kv:(kv[1], kv[0])))

#print(pr)