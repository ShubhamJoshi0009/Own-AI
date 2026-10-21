# VectorDB — Build a Vector Database from Scratch in Java

A fully working **Vector Database** built from scratch in Java with a rich web UI.  
Implements **HNSW**, **KD-Tree**, and **Brute Force** search algorithms side-by-side, plus a **RAG pipeline** powered by a local LLM via Ollama.

> Built as an educational project to show how production vector databases like Pinecone, Weaviate, and Chroma actually work under the hood.

---

## What This Project Does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |
| **Zero External Jars** | Runs purely on standard Java (JDK 21+) using built-in `HttpServer` and `HttpClient` |

---

## How It Works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Java)                  ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph where each layer is progressively sparser — searches start at the top layer and zoom in, achieving $O(\log N)$ complexity instead of $O(N)$ for brute force.

---

## Prerequisites

You need **3 things** installed on your laptop:

1. **Java JDK 21+** (or JDK 17+)
2. **Git**
3. **Ollama** (runs the local AI models)

Verify Java is installed by opening PowerShell or CMD:
```powershell
java -version
javac -version
```

---

## Step-by-Step Setup (Windows)

### Step 1 — Install Ollama (Local AI Models)

1. Go to **https://ollama.com** and download the Windows installer.
2. Run the installer (Ollama starts in the system tray).
3. Open **PowerShell** and pull the models:

```powershell
ollama pull nomic-embed-text
```
*(~274 MB — this is the embedding model)*

```powershell
ollama pull llama3.2
```
*(~2 GB — this is the language model)*

4. Verify Ollama is running:
```powershell
ollama list
```

### Step 2 — Compile and Run the Java Server

You can run the project using any of these simple methods:

#### Option A: One-click script (Recommended)
Double-click `run.bat` or in PowerShell execute:
```powershell
.\run.ps1
```

#### Option B: Standard `javac` and `java`
Inside the project folder, run:
```powershell
javac *.java
java Main
```

#### Option C: Maven (Optional)
If you prefer Maven:
```powershell
mvn compile
mvn exec:java -Dexec.mainClass="Main"
```

You should see:
```
=== VectorDB Engine (Java) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

### Step 3 — Open the Web UI

Open your browser and navigate to:
```
http://localhost:8080
```

---

## Using the Application

### Tab 1: Search (Demo Vectors)

- Type any concept in the search box: `binary tree`, `sushi`, `basketball`, `calculus`
- Choose your algorithm: **HNSW**, **KD-Tree**, or **Brute Force**
- Choose distance metric: **Cosine**, **Euclidean**, or **Manhattan**
- Click **⚡ SEARCH** — results appear with distances, the matching point glows on the scatter plot
- Click **▶ COMPARE ALL ALGOS** to run all 3 algorithms and compare their speed in microseconds

**The scatter plot** shows all 20 vectors projected to 2D using PCA. Notice how the 4 semantic categories (CS, Math, Food, Sports) form distinct clusters.

### Tab 2: Documents (Real Embeddings)

This uses Ollama to generate **real 768-dimensional embeddings** from any text.

1. Type a title (e.g., `Operating Systems Notes`)
2. Paste any text — lecture notes, textbook paragraphs, Wikipedia articles
3. Click **⚡ EMBED & INSERT**
4. Long documents are automatically split into overlapping 250-word chunks (with 30-word overlap)
5. Each chunk gets its own embedding and is stored in a separate HNSW index

### Tab 3: Ask AI (RAG Pipeline)

1. Insert documents in Tab 2 first
2. Type a question about your documents
3. Click **🤖 ASK AI**

What happens behind the scenes:
```
1. Your question → embedded with nomic-embed-text (768D vector)
2. HNSW search → finds 3 most semantically similar chunks (cosine distance <= 0.7)
3. Retrieved chunks → sent as context to llama3.2
4. llama3.2 → generates an answer based on your documents
```

The answer streams into the chat view. Click the **context chips** to inspect the retrieved passages.

---

## REST API Reference

The server exposes a full REST API at `http://localhost:8080`.

### Demo Vector Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete vector by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |

### Document & RAG Endpoints

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` | Embed and store document |
| `GET` | `/doc/list` | — | List all stored documents |
| `DELETE` | `/doc/delete/:id` | — | Delete document chunk |
| `POST` | `/doc/search` | `{"question":"...","k":3}` | Fast retrieval for UI |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` | RAG: retrieve + generate |
| `GET` | `/status` | — | Ollama status and model info |

### Example: Search via curl / PowerShell

```powershell
curl "http://localhost:8080/search?v=0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1&k=3&metric=cosine&algo=hnsw"
```

### Example: Ask a question via curl / PowerShell

```powershell
curl -X POST http://localhost:8080/doc/ask `
  -H "Content-Type: application/json" `
  -d '{"question":"What is dynamic programming?","k":3}'
```

---

## Project Structure

```
VectorDB/
├── Main.java               ← HTTP server (built-in HttpServer), router, and demo data
├── VectorDB.java           ← Thread-safe multi-index manager (HNSW + KD-Tree + Brute Force)
├── DocumentDB.java         ← Document chunk store with HNSW index for high-dimensional embeddings
├── HNSW.java               ← Hierarchical Navigable Small World multilayer graph index
├── KDTree.java             ← Multi-dimensional binary search tree
├── BruteForce.java         ← Exact linear scan baseline
├── DistanceMetrics.java    ← Cosine, Euclidean, and Manhattan distance functions
├── DistFn.java             ← Functional interface for vector distance calculations
├── DistId.java             ← (Distance, ID) pair model for KNN heaps
├── VectorItem.java         ← Vector item model (id, metadata, category, embedding)
├── DocItem.java            ← Document chunk item model
├── DocHit.java             ← Search hit model for document retrieval
├── TextChunker.java        ← Sliding-window chunker with word overlap
├── OllamaClient.java       ← HTTP client for Ollama embeddings and generation
├── JsonUtil.java           ← Zero-dependency JSON parser and serializer
├── TestEngine.java         ← Verification test suite for all algorithms
├── index.html              ← Single-page application frontend
├── run.bat                 ← Windows batch launcher
├── run.ps1                 ← Windows PowerShell launcher
├── pom.xml                 ← Optional Maven build file
└── cpp_original/           ← Original C++ source code kept for reference
```

---

## Algorithm Deep Dive

### 1. HNSW (Hierarchical Navigable Small World)

HNSW constructs a multilayer geometric graph:
- **Layer distribution**: Layer 0 contains all vectors. Each higher layer contains an exponentially smaller fraction of vectors ($P \sim \exp(-l)$).
- **Heuristic descent**: Searches start from the top entry point, greedily moving toward the closest neighbor in the current layer until reaching a local minimum, then stepping down one layer.
- **Layer 0 Beam Search**: At Layer 0, a priority queue maintains a candidate pool of size $ef$, finding high-recall approximate nearest neighbors in $O(\log N)$ time.
- **Edge selection**: During insertion, edges are formed bidirectionally with the top-$M$ closest neighbors.

### 2. KD-Tree (K-Dimensional Tree)

A spatial partitioning tree where:
- Each node partitions space along one coordinate axis (`axis = depth % dimensions`).
- Points with coordinate values less than the partition coordinate branch to the left; others branch to the right.
- Pruning: KNN searches use a bounding box check (`|query[axis] - node[axis]| < heap.maxDist`). If the distance to the splitting hyperplane exceeds the worst distance in the current top-k heap, the entire opposite subtree is pruned.

### 3. Brute Force

Computes the exact distance to every vector in the dataset and sorts the results. Provides a ground-truth baseline ($100\%$ recall) to evaluate index performance and accuracy.

---

## Common Issues & Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama: OFFLINE` in header | Run `ollama serve` in a terminal or check system tray |
| Embedding takes a while | Ollama is downloading the model on first use |
| Port 8080 already in use | Kill process using port 8080: `netstat -ano \| findstr 8080` then `taskkill /PID <pid> /F` |
| LLM answer is slow | llama3.2 takes 10–30s on laptop CPU. Switch to `llama3.2:1b` for faster responses |

### Use a Smaller/Faster LLM

If `llama3.2` is too heavy for your machine, pull the lightweight 1B model:
```powershell
ollama pull llama3.2:1b
```
And in `OllamaClient.java`, update:
```java
public String genModel = "llama3.2:1b";
```
Recompile (`javac *.java`) and restart!

---

## License

MIT — feel free to use and modify!
#   O w n - A I  
 