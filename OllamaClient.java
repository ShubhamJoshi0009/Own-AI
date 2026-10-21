import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;

public class OllamaClient {
    private final String baseUrl;
    private final HttpClient client;
    public String embedModel = "nomic-embed-text";
    public String genModel = "llama3.2";
    public boolean fallbackToSimulation = true;

    public OllamaClient(String host, int port) {
        this.baseUrl = "http://" + host + ":" + port;
        this.client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3))
            .build();
    }

    public OllamaClient() {
        this("127.0.0.1", 11434);
    }

    public boolean isAvailable() {
        try {
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/api/tags"))
                .timeout(Duration.ofSeconds(2))
                .GET()
                .build();
            HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    public float[] embed(String text) {
        // Try real Ollama first
        try {
            String body = "{\"model\":\"" + JsonUtil.escape(embedModel) + "\",\"prompt\":\"" + JsonUtil.escape(text) + "\"}";
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/api/embeddings"))
                .timeout(Duration.ofSeconds(30))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();
            HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() == 200) {
                float[] emb = JsonUtil.extractFloatArray(resp.body(), "embedding");
                if (emb.length > 0) return emb;
            }
        } catch (Exception ignored) {}

        // If offline and fallback enabled, generate deterministic 768D pseudo-semantic embedding
        if (fallbackToSimulation) {
            return generateSimulatedEmbedding(text, 768);
        }
        return new float[0];
    }

    public String generate(String prompt) {
        // Try real Ollama first
        try {
            String body = "{\"model\":\"" + JsonUtil.escape(genModel) + "\","
                + "\"prompt\":\"" + JsonUtil.escape(prompt) + "\","
                + "\"stream\":false}";
            HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/api/generate"))
                .timeout(Duration.ofSeconds(180))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();
            HttpResponse<String> resp = client.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() == 200) {
                String ans = JsonUtil.extractStr(resp.body(), "response");
                if (!ans.isBlank()) return ans;
            }
        } catch (Exception ignored) {}

        // If offline and fallback enabled, generate simulated extractive answer
        if (fallbackToSimulation) {
            return generateSimulatedAnswer(prompt);
        }
        return "ERROR: Ollama unavailable. Run: ollama serve";
    }

    public static float[] generateSimulatedEmbedding(String text, int dims) {
        float[] emb = new float[dims];
        if (text == null || text.isBlank()) return emb;
        String clean = text.toLowerCase().trim();
        String[] words = clean.split("\\s+");
        for (String w : words) {
            if (w.isEmpty()) continue;
            int hash = w.hashCode();
            for (int i = 0; i < 8; i++) {
                int idx = Math.abs((hash ^ (i * 0x9e3779b9)) % dims);
                float weight = 1.0f / (1.0f + (float) Math.log(1 + w.length()));
                emb[idx] += weight;
            }
        }
        // Character trigrams
        for (int i = 0; i <= clean.length() - 3; i++) {
            String trigram = clean.substring(i, i + 3);
            int thash = trigram.hashCode();
            int idx = Math.abs((thash ^ 0x85ebca6b) % dims);
            emb[idx] += 0.35f;
        }
        // L2 Normalize
        float sumSq = 0;
        for (float v : emb) sumSq += v * v;
        if (sumSq > 1e-9f) {
            float norm = (float) Math.sqrt(sumSq);
            for (int i = 0; i < dims; i++) emb[i] /= norm;
        } else {
            emb[0] = 1.0f;
        }
        return emb;
    }

    private static String generateSimulatedAnswer(String prompt) {
        int ctxIdx = prompt.indexOf("Context:\n");
        int qIdx = prompt.indexOf("Question: ");
        int aIdx = prompt.indexOf("Answer:");
        String question = (qIdx != -1 && aIdx != -1) ? prompt.substring(qIdx + 10, aIdx).trim() : "";
        String context = (ctxIdx != -1 && qIdx != -1) ? prompt.substring(ctxIdx + 9, qIdx).trim() : "";

        StringBuilder sb = new StringBuilder();
        sb.append("Based on the retrieved document context");
        if (!question.isEmpty()) {
            sb.append(" for '").append(question).append("'");
        }
        sb.append(":\n\n");

        if (!context.isBlank()) {
            String[] chunks = context.split("\n\n");
            int added = 0;
            for (String chunk : chunks) {
                String trimmed = chunk.trim();
                if (!trimmed.isEmpty()) {
                    sb.append("• ").append(trimmed.replaceAll("^\\[\\d+\\]\\s*", "")).append("\n\n");
                    added++;
                    if (added >= 3) break;
                }
            }
        } else {
            sb.append("No directly matching documents were found in the index for this query.\n\n");
        }
        sb.append("[Simulation Mode Active: Real Ollama is currently offline. To enable neural responses with llama3.2, install Ollama from https://ollama.com and run 'ollama serve']");
        return sb.toString().trim();
    }
}
