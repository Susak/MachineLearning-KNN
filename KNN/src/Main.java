import com.ifmo.ml.knn.KNNLearner;

public class Main {
    public static void main(String[] args) {
        KNNLearner machine = new KNNLearner("chips.txt");
        double min = 2, max = 0, sum = 0;
        for (int i = 0; i < 100; i++) {
            double l = machine.learn();
            sum += l;
            min = Math.min(min, l);
            max = Math.max(max, l);
        }

        System.out.println("min: " + min);
        System.out.println("max: " + max);
        System.out.println("ave: " + sum / 100);
    }
}
