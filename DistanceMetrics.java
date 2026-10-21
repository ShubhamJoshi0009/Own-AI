public final class DistanceMetrics {
    private DistanceMetrics() {}

    public static float euclidean(float[] a, float[] b) {
        float s = 0;
        int len = Math.min(a.length, b.length);
        for (int i = 0; i < len; i++) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return (float) Math.sqrt(s);
    }

    public static float cosine(float[] a, float[] b) {
        float dot = 0, na = 0, nb = 0;
        int len = Math.min(a.length, b.length);
        for (int i = 0; i < len; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na < 1e-9f || nb < 1e-9f) return 1.0f;
        return 1.0f - dot / (float) (Math.sqrt(na) * Math.sqrt(nb));
    }

    public static float manhattan(float[] a, float[] b) {
        float s = 0;
        int len = Math.min(a.length, b.length);
        for (int i = 0; i < len; i++) {
            s += Math.abs(a[i] - b[i]);
        }
        return s;
    }

    public static DistFn getDistFn(String m) {
        if ("cosine".equalsIgnoreCase(m)) return DistanceMetrics::cosine;
        if ("manhattan".equalsIgnoreCase(m)) return DistanceMetrics::manhattan;
        return DistanceMetrics::euclidean;
    }
}
