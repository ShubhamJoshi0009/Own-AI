import java.util.List;

public class TestEngine {
    public static void main(String[] args) {
        System.out.println("Running VectorDB Test Suite...");

        // Test 1: Distance Metrics
        float[] a = new float[]{1.0f, 0.0f, 0.0f};
        float[] b = new float[]{0.0f, 1.0f, 0.0f};
        float euc = DistanceMetrics.euclidean(a, b);
        float cos = DistanceMetrics.cosine(a, b);
        float man = DistanceMetrics.manhattan(a, b);

        assert Math.abs(euc - Math.sqrt(2)) < 1e-4 : "Euclidean failed";
        assert Math.abs(cos - 1.0f) < 1e-4 : "Cosine orthogonal failed";
        assert Math.abs(man - 2.0f) < 1e-4 : "Manhattan failed";

        float[] same = new float[]{0.5f, 0.5f, 0.5f};
        float cosSame = DistanceMetrics.cosine(same, same);
        assert Math.abs(cosSame - 0.0f) < 1e-4 : "Cosine identical failed";

        System.out.println("[PASS] Distance metrics verified");

        // Test 2: Text Chunker
        String sampleText = "one two three four five six seven eight nine ten";
        List<String> chunks = TextChunker.chunkText(sampleText, 5, 2);
        assert chunks.size() >= 2 : "Chunking failed";
        System.out.println("[PASS] TextChunker verified: " + chunks.size() + " chunks");

        // Test 3: VectorDB with Demo Data
        VectorDB db = new VectorDB(16);
        Main.loadDemo(db);
        assert db.size() == 20 : "Demo data load size mismatch: " + db.size();

        float[] query = new float[]{0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f};
        VectorDB.SearchOut resHnsw = db.search(query, 3, "cosine", "hnsw");
        assert resHnsw.hits.size() == 3 : "HNSW search hits mismatch";
        assert resHnsw.hits.get(0).cat.equals("cs") : "HNSW top hit should be cs";

        VectorDB.SearchOut resBf = db.search(query, 3, "cosine", "bruteforce");
        assert resBf.hits.size() == 3 : "BF search hits mismatch";
        assert resBf.hits.get(0).cat.equals("cs") : "BF top hit should be cs";

        VectorDB.SearchOut resKd = db.search(query, 3, "cosine", "kdtree");
        assert resKd.hits.size() == 3 : "KD-Tree search hits mismatch";
        assert resKd.hits.get(0).cat.equals("cs") : "KD-Tree top hit should be cs";

        System.out.println("[PASS] Search algorithms verified across HNSW, KD-Tree, BruteForce");

        // Test 4: Benchmark
        VectorDB.BenchOut bench = db.benchmark(query, 5, "cosine");
        assert bench.n == 20 : "Benchmark item count mismatch";
        System.out.println("[PASS] Benchmark verified: BF=" + bench.bfUs + "us, KD=" + bench.kdUs + "us, HNSW=" + bench.hnswUs + "us");

        // Test 5: Graph Info
        HNSW.GraphInfo info = db.hnswInfo();
        assert info.nodeCount == 20 : "GraphInfo nodeCount mismatch";
        assert info.nodesPerLayer.length > 0 : "GraphInfo nodesPerLayer empty";
        System.out.println("[PASS] HNSW GraphInfo verified: topLayer=" + info.topLayer + ", totalEdges=" + info.edges.size());

        // Test 6: DocumentDB
        DocumentDB docDB = new DocumentDB();
        int docId1 = docDB.insert("Doc 1", "Java vector database architecture", new float[]{0.1f, 0.2f, 0.3f});
        int docId2 = docDB.insert("Doc 2", "Machine learning embeddings and RAG", new float[]{0.1f, 0.22f, 0.31f});
        assert docDB.size() == 2 : "DocDB size mismatch";
        List<DocHit> docHits = docDB.search(new float[]{0.1f, 0.2f, 0.3f}, 2, 0.7f);
        assert !docHits.isEmpty() : "DocDB search should return hits";
        assert docHits.get(0).item.id == docId1 : "DocDB top hit mismatch";

        boolean delOk = docDB.remove(docId1);
        assert delOk : "DocDB deletion failed";
        assert docDB.size() == 1 : "DocDB size after deletion mismatch";
        System.out.println("[PASS] DocumentDB verified");

        // Test 7: JsonUtil
        String json = "{\"metadata\":\"test\\\"quote\", \"category\":\"cs\", \"embedding\":[0.1, 0.2, 0.3], \"val\": 42}";
        String meta = JsonUtil.extractStr(json, "metadata");
        assert meta.equals("test\"quote") : "JsonUtil extractStr failed: " + meta;
        int val = JsonUtil.extractInt(json, "val", 0);
        assert val == 42 : "JsonUtil extractInt failed: " + val;
        float[] arr = JsonUtil.extractFloatArray(json, "embedding");
        assert arr.length == 3 && Math.abs(arr[1] - 0.2f) < 1e-4 : "JsonUtil extractFloatArray failed";
        System.out.println("[PASS] JsonUtil verified");

        System.out.println("\nAll 7 test suites passed successfully!");
    }
}
