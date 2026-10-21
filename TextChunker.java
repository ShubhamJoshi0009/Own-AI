import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TextChunker {
    public static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        if (text == null || text.isBlank()) return Collections.emptyList();
        String[] rawWords = text.trim().split("\\s+");
        List<String> words = new ArrayList<>();
        for (String w : rawWords) {
            if (!w.isEmpty()) words.add(w);
        }
        if (words.isEmpty()) return Collections.emptyList();
        if (words.size() <= chunkWords) {
            return List.of(text.trim());
        }

        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.size(); i += step) {
            int end = Math.min(i + chunkWords, words.size());
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) sb.append(' ');
                sb.append(words.get(j));
            }
            chunks.add(sb.toString());
            if (end == words.size()) break;
        }
        return chunks;
    }

    public static List<String> chunkText(String text) {
        return chunkText(text, 250, 30);
    }
}
