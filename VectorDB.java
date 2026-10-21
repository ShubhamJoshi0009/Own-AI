import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

public class VectorDB {
    public static class Hit {
        public int id;
        public String meta;
        public String cat;
        public float[] emb;
        public float dist;
    }

    public static class SearchOut {
        public List<Hit> hits = new ArrayList<>();
        public long us;
        public String algo;
        public String metric;
    }

    public static class BenchOut {
        public long bfUs, kdUs, hnswUs;
        public int n;
    }

    private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw;
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;
    public final int dims;

    public VectorDB(int d) {
        this.dims = d;
        this.kdt = new KDTree(d);
        this.hnsw = new HNSW(16, 200);
    }

    public int insert(String meta, String cat, float[] emb, DistFn dist) {
        lock.lock();
        try {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return v.id;
        } finally {
            lock.unlock();
        }
    }

    public boolean remove(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            kdt.rebuild(new ArrayList<>(store.values()));
            return true;
        } finally {
            lock.unlock();
        }
    }

    public SearchOut search(float[] q, int k, String metric, String algo) {
        lock.lock();
        try {
            DistFn dfn = DistanceMetrics.getDistFn(metric);
            long t0 = System.nanoTime();

            List<DistId> raw;
            if ("bruteforce".equalsIgnoreCase(algo)) {
                raw = bf.knn(q, k, dfn);
            } else if ("kdtree".equalsIgnoreCase(algo)) {
                raw = kdt.knn(q, k, dfn);
            } else {
                raw = hnsw.knn(q, k, 50, dfn);
            }

            long us = (System.nanoTime() - t0) / 1000;

            SearchOut out = new SearchOut();
            out.us = us;
            out.algo = algo;
            out.metric = metric;

            for (DistId di : raw) {
                VectorItem item = store.get(di.id);
                if (item != null) {
                    Hit hit = new Hit();
                    hit.id = item.id;
                    hit.meta = item.metadata;
                    hit.cat = item.category;
                    hit.emb = item.emb;
                    hit.dist = di.dist;
                    out.hits.add(hit);
                }
            }
            return out;
        } finally {
            lock.unlock();
        }
    }

    public BenchOut benchmark(float[] q, int k, String metric) {
        lock.lock();
        try {
            DistFn dfn = DistanceMetrics.getDistFn(metric);

            long t0 = System.nanoTime();
            bf.knn(q, k, dfn);
            long bfUs = (System.nanoTime() - t0) / 1000;

            long t1 = System.nanoTime();
            kdt.knn(q, k, dfn);
            long kdUs = (System.nanoTime() - t1) / 1000;

            long t2 = System.nanoTime();
            hnsw.knn(q, k, 50, dfn);
            long hnswUs = (System.nanoTime() - t2) / 1000;

            BenchOut out = new BenchOut();
            out.bfUs = bfUs;
            out.kdUs = kdUs;
            out.hnswUs = hnswUs;
            out.n = store.size();
            return out;
        } finally {
            lock.unlock();
        }
    }

    public List<VectorItem> all() {
        lock.lock();
        try {
            return new ArrayList<>(store.values());
        } finally {
            lock.unlock();
        }
    }

    public HNSW.GraphInfo hnswInfo() {
        lock.lock();
        try {
            return hnsw.getInfo();
        } finally {
            lock.unlock();
        }
    }

    public int size() {
        lock.lock();
        try {
            return store.size();
        } finally {
            lock.unlock();
        }
    }
}
