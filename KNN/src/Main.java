import com.ifmo.ml.knn.KNNLearner;
import com.ifmo.ml.knn.utils.Pair;

public class Main {
    public static void main(String[] args) {
        KNNLearner machine = new KNNLearner("chips.txt");
        double min = 2, max = 0, sum = 0;
        for (int i = 0; i < 100; i++) {
            Pair<Double, Double> l = machine.learn();
            sum += l.getFirst();
            min = Math.min(min, l.getFirst());
            max = Math.max(max, l.getFirst());
        }

        System.out.println("min: " + min);
        System.out.println("max: " + max);
        System.out.println("ave: " + sum / 100);
    }
}
