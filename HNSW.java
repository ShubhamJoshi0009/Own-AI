import java.util.*;

public class HNSW {
    public static class Node {
        public final VectorItem item;
        public final int maxLyr;
        public final List<List<Integer>> nbrs;

        public Node(VectorItem item, int maxLyr) {
            this.item = item;
            this.maxLyr = maxLyr;
            this.nbrs = new ArrayList<>(maxLyr + 1);
            for (int i = 0; i <= maxLyr; i++) {
                this.nbrs.add(new ArrayList<>());
            }
        }
    }

    public static class NodeView {
        public final int id;
        public final String metadata;
        public final String category;
        public final int maxLyr;

        public NodeView(int id, String metadata, String category, int maxLyr) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.maxLyr = maxLyr;
        }
    }

    public static class EdgeView {
        public final int src;
        public final int dst;
        public final int lyr;

        public EdgeView(int src, int dst, int lyr) {
            this.src = src;
            this.dst = dst;
            this.lyr = lyr;
        }
    }

    public static class GraphInfo {
        public int topLayer;
        public int nodeCount;
        public int[] nodesPerLayer;
        public int[] edgesPerLayer;
        public final List<NodeView> nodes = new ArrayList<>();
        public final List<EdgeView> edges = new ArrayList<>();
    }

    private final Map<Integer, Node> G = new LinkedHashMap<>();
    private final int M;
    private final int M0;
    private final int ef_build;
    private final float mL;
    private int topLayer = -1;
    private int entryPt = -1;
    private final Random rng;

    public HNSW(int m, int efBuild) {
        this.M = m;
        this.M0 = 2 * m;
        this.ef_build = efBuild;
        this.mL = 1.0f / (float) Math.log(m);
        this.rng = new Random(42);
    }

    public HNSW() {
        this(16, 200);
    }

    private int randLevel() {
        double u = rng.nextDouble();
        if (u <= 0.0) u = 1e-9;
        return (int) Math.floor(-Math.log(u) * mL);
    }

    private List<DistId> searchLayer(float[] q, int ep, int ef, int lyr, DistFn dist) {
        Set<Integer> vis = new HashSet<>();
        // min-heap: candidate pool to expand next (smallest distance first)
        PriorityQueue<DistId> cands = new PriorityQueue<>(Comparator.comparingDouble(p -> p.dist));
        // max-heap: current best elements found so far (largest distance at top)
        PriorityQueue<DistId> found = new PriorityQueue<>((a, b) -> Float.compare(b.dist, a.dist));

        Node epNode = G.get(ep);
        if (epNode == null) return Collections.emptyList();

        float d0 = dist.apply(q, epNode.item.emb);
        vis.add(ep);
        cands.add(new DistId(d0, ep));
        found.add(new DistId(d0, ep));

        while (!cands.isEmpty()) {
            DistId curr = cands.poll();
            float cd = curr.dist;
            int cid = curr.id;

            if (found.size() >= ef && cd > found.peek().dist) {
                break;
            }

            Node cNode = G.get(cid);
            if (cNode == null || lyr >= cNode.nbrs.size()) continue;

            for (int nid : cNode.nbrs.get(lyr)) {
                if (vis.contains(nid) || !G.containsKey(nid)) continue;
                vis.add(nid);

                Node nNode = G.get(nid);
                float nd = dist.apply(q, nNode.item.emb);

                if (found.size() < ef || nd < found.peek().dist) {
                    cands.add(new DistId(nd, nid));
                    found.add(new DistId(nd, nid));
                    if (found.size() > ef) {
                        found.poll();
                    }
                }
            }
        }

        List<DistId> res = new ArrayList<>(found);
        res.sort(Comparator.comparingDouble(p -> p.dist));
        return res;
    }

    private List<Integer> selectNbrs(List<DistId> cands, int maxM) {
        List<Integer> r = new ArrayList<>();
        int limit = Math.min(cands.size(), maxM);
        for (int i = 0; i < limit; i++) {
            r.add(cands.get(i).id);
        }
        return r;
    }

    public void insert(VectorItem item, DistFn dist) {
        int id = item.id;
        int lvl = randLevel();
        Node node = new Node(item, lvl);
        G.put(id, node);

        if (entryPt == -1) {
            entryPt = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            Node epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<DistId> W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = W.get(0).id;
            }
        }

        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<DistId> W = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> sel = selectNbrs(W, maxM);
            node.nbrs.set(lc, new ArrayList<>(sel));

            for (int nid : sel) {
                Node nNode = G.get(nid);
                if (nNode == null) continue;
                while (nNode.nbrs.size() <= lc) {
                    nNode.nbrs.add(new ArrayList<>());
                }
                List<Integer> conn = nNode.nbrs.get(lc);
                conn.add(id);
                if (conn.size() > maxM) {
                    List<DistId> ds = new ArrayList<>();
                    for (int c : conn) {
                        if (G.containsKey(c)) {
                            ds.add(new DistId(dist.apply(nNode.item.emb, G.get(c).item.emb), c));
                        }
                    }
                    ds.sort(Comparator.comparingDouble(p -> p.dist));
                    conn.clear();
                    int limit = Math.min(maxM, ds.size());
                    for (int i = 0; i < limit; i++) {
                        conn.add(ds.get(i).id);
                    }
                }
            }

            if (!W.isEmpty()) ep = W.get(0).id;
        }

        if (lvl > topLayer) {
            topLayer = lvl;
            entryPt = id;
        }
    }

    public List<DistId> knn(float[] q, int k, int ef, DistFn dist) {
        if (entryPt == -1) return Collections.emptyList();
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<DistId> W = searchLayer(q, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = W.get(0).id;
            }
        }
        List<DistId> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        if (W.size() > k) {
            return new ArrayList<>(W.subList(0, k));
        }
        return W;
    }

    public void remove(int id) {
        if (!G.containsKey(id)) return;
        for (Node nd : G.values()) {
            for (List<Integer> layer : nd.nbrs) {
                layer.removeIf(nbrId -> nbrId == id);
            }
        }
        if (entryPt == id) {
            entryPt = -1;
            for (int nid : G.keySet()) {
                if (nid != id) {
                    entryPt = nid;
                    break;
                }
            }
        }
        G.remove(id);
    }

    public GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer = topLayer;
        gi.nodeCount = G.size();
        int maxL = Math.max(topLayer + 1, 1);
        gi.nodesPerLayer = new int[maxL];
        gi.edgesPerLayer = new int[maxL];

        for (Map.Entry<Integer, Node> entry : G.entrySet()) {
            int id = entry.getKey();
            Node nd = entry.getValue();
            gi.nodes.add(new NodeView(id, nd.item.metadata, nd.item.category, nd.maxLyr));
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.add(new EdgeView(id, nid, lc));
                        }
                    }
                }
            }
        }
        return gi;
    }

    public int size() {
        return G.size();
    }
}
