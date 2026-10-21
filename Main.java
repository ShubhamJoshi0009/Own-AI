import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Main {
    private static final int DIMS = 16;
    private static final Pattern DELETE_ITEM_PATTERN = Pattern.compile("^/delete/(\\d+)$");
    private static final Pattern DELETE_DOC_PATTERN = Pattern.compile("^/doc/delete/(\\d+)$");

    public static void main(String[] args) throws IOException {
        VectorDB db = new VectorDB(DIMS);
        DocumentDB docDB = new DocumentDB();
        OllamaClient ollama = new OllamaClient();

        loadDemo(db);

        boolean ollamaUp = ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java) ===");
        System.out.println("http://localhost:8080");
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + ollama.embedModel + "  gen model: " + ollama.genModel);
        }

        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", 8080), 0);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());

        server.createContext("/", new HttpHandler() {
            @Override
            public void handle(HttpExchange exchange) throws IOException {
                handleRequest(exchange, db, docDB, ollama);
            }
        });

        server.start();
    }

    private static void handleRequest(HttpExchange exchange, VectorDB db, DocumentDB docDB, OllamaClient ollama) throws IOException {
        String method = exchange.getRequestMethod().toUpperCase();
        String path = exchange.getRequestURI().getPath();

        // Handle CORS Preflight
        if ("OPTIONS".equals(method)) {
            sendCorsResponse(exchange, 204);
            return;
        }

        try {
            // Serve frontend
            if (("GET".equals(method)) && (path.equals("/") || path.equals("/index.html"))) {
                serveFile(exchange, "index.html", "text/html; charset=utf-8");
                return;
            }

            // GET /search?v=...&k=...&metric=...&algo=...
            if ("GET".equals(method) && path.equals("/search")) {
                Map<String, String> params = parseQueryParams(exchange.getRequestURI().getRawQuery());
                float[] q = JsonUtil.parseVec(params.get("v"));
                if (q.length != DIMS) {
                    sendJsonResponse(exchange, 200, "{\"error\":\"need " + DIMS + "D vector\"}");
                    return;
                }
                int k = 5;
                try {
                    if (params.containsKey("k")) k = Integer.parseInt(params.get("k"));
                } catch (Exception ignored) {}
                String metric = params.getOrDefault("metric", "cosine");
                String algo = params.getOrDefault("algo", "hnsw");

                VectorDB.SearchOut out = db.search(q, k, metric, algo);
                StringBuilder ss = new StringBuilder();
                ss.append("{\"results\":[");
                for (int i = 0; i < out.hits.size(); i++) {
                    if (i > 0) ss.append(',');
                    VectorDB.Hit h = out.hits.get(i);
                    ss.append("{\"id\":").append(h.id)
                      .append(",\"metadata\":").append(JsonUtil.quote(h.meta))
                      .append(",\"category\":").append(JsonUtil.quote(h.cat))
                      .append(",\"distance\":").append(String.format(Locale.US, "%.6f", h.dist))
                      .append(",\"embedding\":").append(JsonUtil.toJson(h.emb)).append('}');
                }
                ss.append("],\"latencyUs\":").append(out.us)
                  .append(",\"algo\":").append(JsonUtil.quote(out.algo))
                  .append(",\"metric\":").append(JsonUtil.quote(out.metric)).append('}');
                sendJsonResponse(exchange, 200, ss.toString());
                return;
            }

            // POST /insert
            if ("POST".equals(method) && path.equals("/insert")) {
                String body = readRequestBody(exchange);
                JsonUtil.InsertRequest req = JsonUtil.parseInsertBody(body);
                if (req.metadata == null || req.metadata.isEmpty() || req.embedding == null || req.embedding.length != DIMS) {
                    sendJsonResponse(exchange, 200, "{\"error\":\"invalid body\"}");
                    return;
                }
                int id = db.insert(req.metadata, req.category, req.embedding, DistanceMetrics.getDistFn("cosine"));
                sendJsonResponse(exchange, 200, "{\"id\":" + id + "}");
                return;
            }

            // DELETE /delete/{id}
            Matcher delMatcher = DELETE_ITEM_PATTERN.matcher(path);
            if ("DELETE".equals(method) && delMatcher.matches()) {
                int id = Integer.parseInt(delMatcher.group(1));
                boolean ok = db.remove(id);
                sendJsonResponse(exchange, 200, "{\"ok\":" + (ok ? "true" : "false") + "}");
                return;
            }

            // GET /items
            if ("GET".equals(method) && path.equals("/items")) {
                List<VectorItem> items = db.all();
                StringBuilder ss = new StringBuilder();
                ss.append('[');
                for (int i = 0; i < items.size(); i++) {
                    if (i > 0) ss.append(',');
                    VectorItem v = items.get(i);
                    ss.append("{\"id\":").append(v.id)
                      .append(",\"metadata\":").append(JsonUtil.quote(v.metadata))
                      .append(",\"category\":").append(JsonUtil.quote(v.category))
                      .append(",\"embedding\":").append(JsonUtil.toJson(v.emb)).append('}');
                }
                ss.append(']');
                sendJsonResponse(exchange, 200, ss.toString());
                return;
            }

            // GET /benchmark?v=...&k=...&metric=...
            if ("GET".equals(method) && path.equals("/benchmark")) {
                Map<String, String> params = parseQueryParams(exchange.getRequestURI().getRawQuery());
                float[] q = JsonUtil.parseVec(params.get("v"));
                if (q.length != DIMS) {
                    sendJsonResponse(exchange, 200, "{\"error\":\"need " + DIMS + "D vector\"}");
                    return;
                }
                int k = 5;
                try {
                    if (params.containsKey("k")) k = Integer.parseInt(params.get("k"));
                } catch (Exception ignored) {}
                String metric = params.getOrDefault("metric", "cosine");
                VectorDB.BenchOut b = db.benchmark(q, k, metric);
                String json = "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs
                    + ",\"hnswUs\":" + b.hnswUs + ",\"itemCount\":" + b.n + "}";
                sendJsonResponse(exchange, 200, json);
                return;
            }

            // GET /hnsw-info
            if ("GET".equals(method) && path.equals("/hnsw-info")) {
                HNSW.GraphInfo gi = db.hnswInfo();
                StringBuilder ss = new StringBuilder();
                ss.append("{\"topLayer\":").append(gi.topLayer)
                  .append(",\"nodeCount\":").append(gi.nodeCount)
                  .append(",\"nodesPerLayer\":[");
                for (int i = 0; i < gi.nodesPerLayer.length; i++) {
                    if (i > 0) ss.append(',');
                    ss.append(gi.nodesPerLayer[i]);
                }
                ss.append("],\"edgesPerLayer\":[");
                for (int i = 0; i < gi.edgesPerLayer.length; i++) {
                    if (i > 0) ss.append(',');
                    ss.append(gi.edgesPerLayer[i]);
                }
                ss.append("],\"nodes\":[");
                for (int i = 0; i < gi.nodes.size(); i++) {
                    if (i > 0) ss.append(',');
                    HNSW.NodeView n = gi.nodes.get(i);
                    ss.append("{\"id\":").append(n.id)
                      .append(",\"metadata\":").append(JsonUtil.quote(n.metadata))
                      .append(",\"category\":").append(JsonUtil.quote(n.category))
                      .append(",\"maxLyr\":").append(n.maxLyr).append('}');
                }
                ss.append("],\"edges\":[");
                for (int i = 0; i < gi.edges.size(); i++) {
                    if (i > 0) ss.append(',');
                    HNSW.EdgeView e = gi.edges.get(i);
                    ss.append("{\"src\":").append(e.src)
                      .append(",\"dst\":").append(e.dst)
                      .append(",\"lyr\":").append(e.lyr).append('}');
                }
                ss.append("]}");
                sendJsonResponse(exchange, 200, ss.toString());
                return;
            }

            // POST /doc/insert
            if ("POST".equals(method) && path.equals("/doc/insert")) {
                String body = readRequestBody(exchange);
                String title = JsonUtil.extractStr(body, "title");
                String text = JsonUtil.extractStr(body, "text");
                if (title.isEmpty() || text.isEmpty()) {
                    sendJsonResponse(exchange, 200, "{\"error\":\"need title and text\"}");
                    return;
                }

                List<String> chunks = TextChunker.chunkText(text, 250, 30);
                List<Integer> ids = new ArrayList<>();
                for (int i = 0; i < chunks.size(); i++) {
                    float[] emb = ollama.embed(chunks.get(i));
                    if (emb.length == 0) {
                        sendJsonResponse(exchange, 200,
                            "{\"error\":\"Ollama unavailable. Install from https://ollama.com then run: "
                            + "ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                        return;
                    }
                    String chunkTitle = (chunks.size() > 1)
                        ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                        : title;
                    ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
                }

                StringBuilder ss = new StringBuilder();
                ss.append("{\"ids\":[");
                for (int i = 0; i < ids.size(); i++) {
                    if (i > 0) ss.append(',');
                    ss.append(ids.get(i));
                }
                ss.append("],\"chunks\":").append(chunks.size())
                  .append(",\"dims\":").append(docDB.getDims()).append('}');
                sendJsonResponse(exchange, 200, ss.toString());
                return;
            }

            // DELETE /doc/delete/{id}
            Matcher delDocMatcher = DELETE_DOC_PATTERN.matcher(path);
            if ("DELETE".equals(method) && delDocMatcher.matches()) {
                int id = Integer.parseInt(delDocMatcher.group(1));
                boolean ok = docDB.remove(id);
                sendJsonResponse(exchange, 200, "{\"ok\":" + (ok ? "true" : "false") + "}");
                return;
            }

            // GET /doc/list
            if ("GET".equals(method) && path.equals("/doc/list")) {
                List<DocItem> docs = docDB.all();
                StringBuilder ss = new StringBuilder();
                ss.append('[');
                for (int i = 0; i < docs.size(); i++) {
                    if (i > 0) ss.append(',');
                    DocItem doc = docs.get(i);
                    String preview = doc.text.length() > 120 ? doc.text.substring(0, 120) + "…" : doc.text;
                    int words = 0;
                    for (int c = 0; c < doc.text.length(); c++) {
                        if (doc.text.charAt(c) == ' ') words++;
                    }
                    words++;
                    ss.append("{\"id\":").append(doc.id)
                      .append(",\"title\":").append(JsonUtil.quote(doc.title))
                      .append(",\"preview\":").append(JsonUtil.quote(preview))
                      .append(",\"words\":").append(words).append('}');
                }
                ss.append(']');
                sendJsonResponse(exchange, 200, ss.toString());
                return;
            }

            // POST /doc/search
            if ("POST".equals(method) && path.equals("/doc/search")) {
                String body = readRequestBody(exchange);
                String question = JsonUtil.extractStr(body, "question");
                int k = JsonUtil.extractInt(body, "k", 3);
                if (question.isEmpty()) {
                    sendJsonResponse(exchange, 200, "{\"error\":\"need question\"}");
                    return;
                }

                float[] qEmb = ollama.embed(question);
                if (qEmb.length == 0) {
                    sendJsonResponse(exchange, 200, "{\"error\":\"Ollama unavailable\"}");
                    return;
                }

                List<DocHit> hits = docDB.search(qEmb, k);
                StringBuilder ss = new StringBuilder();
                ss.append("{\"contexts\":[");
                for (int i = 0; i < hits.size(); i++) {
                    if (i > 0) ss.append(',');
                    DocHit hit = hits.get(i);
                    ss.append("{\"id\":").append(hit.item.id)
                      .append(",\"title\":").append(JsonUtil.quote(hit.item.title))
                      .append(",\"distance\":").append(String.format(Locale.US, "%.4f", hit.dist)).append('}');
                }
                ss.append("]}");
                sendJsonResponse(exchange, 200, ss.toString());
                return;
            }

            // POST /doc/ask
            if ("POST".equals(method) && path.equals("/doc/ask")) {
                String body = readRequestBody(exchange);
                String question = JsonUtil.extractStr(body, "question");
                int k = JsonUtil.extractInt(body, "k", 3);
                if (question.isEmpty()) {
                    sendJsonResponse(exchange, 200, "{\"error\":\"need question\"}");
                    return;
                }

                float[] qEmb = ollama.embed(question);
                if (qEmb.length == 0) {
                    sendJsonResponse(exchange, 200, "{\"error\":\"Ollama unavailable\"}");
                    return;
                }

                List<DocHit> hits = docDB.search(qEmb, k);
                StringBuilder ctx = new StringBuilder();
                for (int i = 0; i < hits.size(); i++) {
                    ctx.append("[").append(i + 1).append("] ").append(hits.get(i).item.title).append(":\n")
                       .append(hits.get(i).item.text).append("\n\n");
                }

                String prompt = "You are a helpful assistant. Answer the user's question directly. "
                    + "Use the provided context if it contains relevant information. "
                    + "If it doesn't, just use your own general knowledge. "
                    + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                    + "Just answer the question naturally.\n\n"
                    + "Context:\n" + ctx.toString()
                    + "Question: " + question + "\n\n"
                    + "Answer:";

                String answer = ollama.generate(prompt);

                StringBuilder ss = new StringBuilder();
                ss.append("{\"answer\":").append(JsonUtil.quote(answer))
                  .append(",\"model\":").append(JsonUtil.quote(ollama.genModel))
                  .append(",\"contexts\":[");
                for (int i = 0; i < hits.size(); i++) {
                    if (i > 0) ss.append(',');
                    DocHit hit = hits.get(i);
                    ss.append("{\"id\":").append(hit.item.id)
                      .append(",\"title\":").append(JsonUtil.quote(hit.item.title))
                      .append(",\"text\":").append(JsonUtil.quote(hit.item.text))
                      .append(",\"distance\":").append(String.format(Locale.US, "%.4f", hit.dist)).append('}');
                }
                ss.append("],\"docCount\":").append(docDB.size()).append('}');
                sendJsonResponse(exchange, 200, ss.toString());
                return;
            }

            // GET /status
            if ("GET".equals(method) && path.equals("/status")) {
                boolean up = ollama.isAvailable();
                String json = "{\"ollamaAvailable\":" + (up ? "true" : "false")
                    + ",\"simulationMode\":" + (!up ? "true" : "false")
                    + ",\"embedModel\":" + JsonUtil.quote(up ? ollama.embedModel : "simulated-768d (offline mode)")
                    + ",\"genModel\":" + JsonUtil.quote(up ? ollama.genModel : "extractive-rag (offline mode)")
                    + ",\"docCount\":" + docDB.size()
                    + ",\"docDims\":" + docDB.getDims()
                    + ",\"demoDims\":" + DIMS
                    + ",\"demoCount\":" + db.size() + "}";
                sendJsonResponse(exchange, 200, json);
                return;
            }

            // GET /stats
            if ("GET".equals(method) && path.equals("/stats")) {
                String json = "{\"count\":" + db.size()
                    + ",\"dims\":" + DIMS
                    + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                    + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
                sendJsonResponse(exchange, 200, json);
                return;
            }

            // 404 fallback
            sendJsonResponse(exchange, 404, "{\"error\":\"Not Found\"}");

        } catch (Exception e) {
            e.printStackTrace();
            sendJsonResponse(exchange, 500, "{\"error\":\"Internal Server Error: " + JsonUtil.escape(e.getMessage()) + "\"}");
        }
    }

    private static Map<String, String> parseQueryParams(String query) {
        Map<String, String> params = new HashMap<>();
        if (query == null || query.isBlank()) return params;
        for (String pair : query.split("&")) {
            int idx = pair.indexOf('=');
            if (idx > 0) {
                String key = URLDecoder.decode(pair.substring(0, idx), StandardCharsets.UTF_8);
                String val = URLDecoder.decode(pair.substring(idx + 1), StandardCharsets.UTF_8);
                params.put(key, val);
            } else if (!pair.isEmpty()) {
                params.put(URLDecoder.decode(pair, StandardCharsets.UTF_8), "");
            }
        }
        return params;
    }

    private static String readRequestBody(HttpExchange exchange) throws IOException {
        try (InputStream is = exchange.getRequestBody()) {
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static void sendCorsResponse(HttpExchange exchange, int statusCode) throws IOException {
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
        exchange.sendResponseHeaders(statusCode, -1);
        exchange.close();
    }

    private static void sendJsonResponse(HttpExchange exchange, int statusCode, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(statusCode, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static void serveFile(HttpExchange exchange, String filePath, String contentType) throws IOException {
        File file = new File(filePath);
        if (!file.exists() || !file.isFile()) {
            sendJsonResponse(exchange, 404, "{\"error\":\"File not found: " + filePath + "\"}");
            return;
        }
        byte[] bytes = Files.readAllBytes(file.toPath());
        exchange.getResponseHeaders().set("Access-Control-Allow-Origin", "*");
        exchange.getResponseHeaders().set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        exchange.getResponseHeaders().set("Access-Control-Allow-Headers", "Content-Type");
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(bytes);
        }
    }

    public static void loadDemo(VectorDB db) {
        DistFn dist = DistanceMetrics.getDistFn("cosine");
        // Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
        db.insert("Linked List: nodes connected by pointers", "cs",
            new float[]{0.90f,0.85f,0.72f,0.68f,0.12f,0.08f,0.15f,0.10f,0.05f,0.08f,0.06f,0.09f,0.07f,0.11f,0.08f,0.06f}, dist);
        db.insert("Binary Search Tree: O(log n) search and insert", "cs",
            new float[]{0.88f,0.82f,0.78f,0.74f,0.15f,0.10f,0.08f,0.12f,0.06f,0.07f,0.08f,0.05f,0.09f,0.06f,0.07f,0.10f}, dist);
        db.insert("Dynamic Programming: memoization overlapping subproblems", "cs",
            new float[]{0.82f,0.76f,0.88f,0.80f,0.20f,0.18f,0.12f,0.09f,0.07f,0.06f,0.08f,0.07f,0.08f,0.09f,0.06f,0.07f}, dist);
        db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs",
            new float[]{0.85f,0.80f,0.75f,0.82f,0.18f,0.14f,0.10f,0.08f,0.06f,0.09f,0.07f,0.06f,0.10f,0.08f,0.09f,0.07f}, dist);
        db.insert("Hash Table: O(1) lookup with collision chaining", "cs",
            new float[]{0.87f,0.78f,0.70f,0.76f,0.13f,0.11f,0.09f,0.14f,0.08f,0.07f,0.06f,0.08f,0.07f,0.10f,0.08f,0.09f}, dist);
        db.insert("Calculus: derivatives integrals and limits", "math",
            new float[]{0.12f,0.15f,0.18f,0.10f,0.91f,0.86f,0.78f,0.72f,0.08f,0.06f,0.07f,0.09f,0.07f,0.08f,0.06f,0.10f}, dist);
        db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math",
            new float[]{0.20f,0.18f,0.15f,0.12f,0.88f,0.90f,0.82f,0.76f,0.09f,0.07f,0.08f,0.06f,0.10f,0.07f,0.08f,0.09f}, dist);
        db.insert("Probability: distributions random variables Bayes theorem", "math",
            new float[]{0.15f,0.12f,0.20f,0.18f,0.84f,0.80f,0.88f,0.82f,0.07f,0.08f,0.06f,0.10f,0.09f,0.06f,0.09f,0.08f}, dist);
        db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math",
            new float[]{0.22f,0.16f,0.14f,0.20f,0.80f,0.85f,0.76f,0.90f,0.08f,0.09f,0.07f,0.06f,0.08f,0.10f,0.07f,0.06f}, dist);
        db.insert("Combinatorics: permutations combinations generating functions", "math",
            new float[]{0.18f,0.20f,0.16f,0.14f,0.86f,0.78f,0.84f,0.80f,0.06f,0.07f,0.09f,0.08f,0.06f,0.09f,0.10f,0.07f}, dist);
        db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.90f,0.86f,0.78f,0.72f,0.08f,0.06f,0.09f,0.07f}, dist);
        db.insert("Sushi: vinegared rice raw fish and nori rolls", "food",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.09f,0.06f,0.08f,0.07f,0.86f,0.90f,0.82f,0.76f,0.07f,0.09f,0.06f,0.08f}, dist);
        db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
            new float[]{0.09f,0.07f,0.06f,0.08f,0.08f,0.09f,0.07f,0.06f,0.82f,0.78f,0.90f,0.84f,0.09f,0.07f,0.08f,0.06f}, dist);
        db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
            new float[]{0.07f,0.09f,0.08f,0.06f,0.06f,0.07f,0.09f,0.08f,0.78f,0.82f,0.86f,0.90f,0.06f,0.08f,0.07f,0.09f}, dist);
        db.insert("Croissant: laminated pastry with buttery flaky layers", "food",
            new float[]{0.06f,0.07f,0.10f,0.09f,0.10f,0.06f,0.07f,0.10f,0.85f,0.80f,0.76f,0.82f,0.09f,0.07f,0.10f,0.06f}, dist);
        db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports",
            new float[]{0.09f,0.07f,0.08f,0.10f,0.08f,0.09f,0.07f,0.06f,0.08f,0.07f,0.09f,0.06f,0.91f,0.85f,0.78f,0.72f}, dist);
        db.insert("Football: tackles touchdowns field goals and strategy", "sports",
            new float[]{0.07f,0.09f,0.06f,0.08f,0.09f,0.07f,0.10f,0.08f,0.07f,0.09f,0.08f,0.07f,0.87f,0.89f,0.82f,0.76f}, dist);
        db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
            new float[]{0.08f,0.06f,0.09f,0.07f,0.07f,0.08f,0.06f,0.09f,0.09f,0.06f,0.07f,0.08f,0.83f,0.80f,0.88f,0.82f}, dist);
        db.insert("Chess: openings endgames tactics strategic board game", "sports",
            new float[]{0.25f,0.20f,0.22f,0.18f,0.22f,0.18f,0.20f,0.15f,0.06f,0.08f,0.07f,0.09f,0.80f,0.84f,0.78f,0.90f}, dist);
        db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
            new float[]{0.06f,0.08f,0.07f,0.09f,0.08f,0.06f,0.09f,0.07f,0.10f,0.08f,0.06f,0.07f,0.85f,0.82f,0.86f,0.80f}, dist);
    }
}
