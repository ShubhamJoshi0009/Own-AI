public class DocItem {
    public final int id;
    public final String title;
    public final String text;
    public final float[] emb;

    public DocItem(int id, String title, String text, float[] emb) {
        this.id = id;
        this.title = title;
        this.text = text;
        this.emb = emb;
    }
}
