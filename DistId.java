public class DistId implements Comparable<DistId> {
    public final float dist;
    public final int id;

    public DistId(float dist, int id) {
        this.dist = dist;
        this.id = id;
    }

    @Override
    public int compareTo(DistId o) {
        return Float.compare(this.dist, o.dist);
    }

    @Override
    public String toString() {
        return "DistId{" + "dist=" + dist + ", id=" + id + '}';
    }
}
