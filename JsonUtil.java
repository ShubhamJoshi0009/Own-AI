import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class JsonUtil {
    private JsonUtil() {}

    public static String escape(String s) {
        if (s == null) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.toString();
    }

    public static String quote(String s) {
        return "\"" + escape(s) + "\"";
    }

    public static String extractStr(String body, String key) {
        if (body == null || key == null) return "";
        String needle = "\"" + key + "\"";
        int p = body.indexOf(needle);
        if (p == -1) return "";
        p = body.indexOf(':', p + needle.length());
        if (p == -1) return "";
        p++;
        while (p < body.length() && Character.isWhitespace(body.charAt(p))) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                char next = body.charAt(p);
                switch (next) {
                    case '"' -> result.append('"');
                    case '\\' -> result.append('\\');
                    case '/' -> result.append('/');
                    case 'b' -> result.append('\b');
                    case 'f' -> result.append('\f');
                    case 'n' -> result.append('\n');
                    case 'r' -> result.append('\r');
                    case 't' -> result.append('\t');
                    case 'u' -> {
                        if (p + 4 < body.length()) {
                            try {
                                int code = Integer.parseInt(body.substring(p + 1, p + 5), 16);
                                result.append((char) code);
                                p += 4;
                            } catch (NumberFormatException ignored) {
                                result.append('u');
                            }
                        } else {
                            result.append('u');
                        }
                    }
                    default -> result.append(next);
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    public static int extractInt(String body, String key, int def) {
        if (body == null || key == null) return def;
        String needle = "\"" + key + "\"";
        int p = body.indexOf(needle);
        if (p == -1) return def;
        p = body.indexOf(':', p + needle.length());
        if (p == -1) return def;
        p++;
        while (p < body.length() && Character.isWhitespace(body.charAt(p))) p++;
        int start = p;
        if (p < body.length() && (body.charAt(p) == '-' || body.charAt(p) == '+')) p++;
        while (p < body.length() && Character.isDigit(body.charAt(p))) p++;
        if (p > start) {
            try {
                return Integer.parseInt(body.substring(start, p));
            } catch (NumberFormatException e) {
                return def;
            }
        }
        return def;
    }

    public static float[] parseVec(String s) {
        if (s == null || s.isBlank()) return new float[0];
        String str = s.trim();
        if (str.startsWith("[")) str = str.substring(1);
        if (str.endsWith("]")) str = str.substring(0, str.length() - 1);
        String[] parts = str.split(",");
        List<Float> list = new ArrayList<>();
        for (String part : parts) {
            String t = part.trim();
            if (!t.isEmpty()) {
                try {
                    list.add(Float.parseFloat(t));
                } catch (NumberFormatException ignored) {}
            }
        }
        float[] res = new float[list.size()];
        for (int i = 0; i < list.size(); i++) res[i] = list.get(i);
        return res;
    }

    public static float[] extractFloatArray(String body, String key) {
        if (body == null || key == null) return new float[0];
        String needle = "\"" + key + "\"";
        int p = body.indexOf(needle);
        if (p == -1) return new float[0];
        p = body.indexOf('[', p + needle.length());
        if (p == -1) return new float[0];
        int depth = 1;
        int e = p + 1;
        while (e < body.length() && depth > 0) {
            if (body.charAt(e) == '[') depth++;
            else if (body.charAt(e) == ']') depth--;
            e++;
        }
        if (depth == 0) {
            return parseVec(body.substring(p + 1, e - 1));
        }
        return new float[0];
    }

    public static String toJson(float[] v) {
        if (v == null) return "[]";
        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(',');
            sb.append(String.format(Locale.US, "%.4f", v[i]));
        }
        sb.append(']');
        return sb.toString();
    }

    public static class InsertRequest {
        public String metadata = "";
        public String category = "";
        public float[] embedding = new float[0];
    }

    public static InsertRequest parseInsertBody(String body) {
        InsertRequest req = new InsertRequest();
        req.metadata = extractStr(body, "metadata");
        req.category = extractStr(body, "category");
        req.embedding = extractFloatArray(body, "embedding");
        return req;
    }
}
