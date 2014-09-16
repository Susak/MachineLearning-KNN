package com.ifmo.ml.knn;

import com.ifmo.ml.knn.dividers.CrossValidationDivider;
import com.ifmo.ml.knn.dividers.Divider;
import com.ifmo.ml.knn.exceptions.DataException;
import com.ifmo.ml.knn.kernelfunctions.KernelFunction;
import com.ifmo.ml.knn.kernelfunctions.TriweightKernelFunction;
import com.ifmo.ml.knn.metrics.*;
import com.ifmo.ml.knn.utils.Pair;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.function.Predicate;

public class KNNLearner {
    private static int KD_FOLD_CONST = 5;
    private static Metric m = new ManhattanMetric();
    private static KernelFunction kf = new TriweightKernelFunction();

    private String dataSetFileName;

    public KNNLearner(String dataSetFileName) {
        this.dataSetFileName = dataSetFileName;
    }

    private List<Precedent> getPrecedents() {
        List<Precedent> precedents = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(dataSetFileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] data = line.split(",");
                precedents.add(
                        new Precedent(
                            Integer.parseInt(data[data.length - 1]),
                            Arrays.stream(data, 0, data.length - 1)
                                  .mapToDouble(Double::parseDouble)
                                  .toArray()
                        )
                );
            }
        } catch (IOException e) {
            System.err.println(e.getMessage());
            return null;
        }

        return precedents;
    }

    private double classSimilarity(Precedent p, List<Precedent> trains,
                                   int k, int eClass
    ) {
        return trains
                .stream()
                .sorted((p1, p2) -> Double.compare(
                        m.countMetric(p1, p), m.countMetric(p2, p)
                ))
                .limit(k)
                .filter(pr -> pr.geteClass() == eClass)
                .mapToDouble(pc -> kf.evaluate(m.countMetric(pc, p) / m.countMetric(trains.get(k), p)))
                .sum();
    }

    private double countKNNDistance(List<Precedent> precedents, int k) {
        int match = 0;
        Divider<Precedent> divider = new CrossValidationDivider<>(precedents, KD_FOLD_CONST);
        for (int i = 0; i < divider.iterations(); i++) {
            divider.divide(i);
            List<Precedent> trains = divider.getTrainingSamples(),
                            tests  = divider.getTestingSamples();

            // learn on trains and test on tests (C)

            for (Precedent p : tests) {
                double c1 = classSimilarity(p, trains, Math.min(k, trains.size() - 1), 0),
                       c2 = classSimilarity(p, trains, Math.min(k, trains.size() - 1), 1);
                if (c1 > c2) {
                    match += (p.geteClass() == 0) ? 1 : 0;
                } else {
                    match += (p.geteClass() == 1) ? 1 : 0;
                }
            }
        }

        return 1.0 * match / divider.getParts();
    }

    private int findOptimalK(List<Precedent> trains) {
        int optK = -1;
        double optAccuracy = 0;
        for (int k = 1; k < trains.size(); k++) {
            double d = countKNNDistance(trains, k);
            if (d > optAccuracy) {
                optAccuracy = d;
                optK = k;
            }
        }
        return optK;
    }

    private Pair<Double, Double> findMeasure(List<Precedent> tests, int optK) {
        int[][] tm = new int[2][2];
        for (Precedent p : tests) {
            double c1 = classSimilarity(p, tests, Math.min(optK, tests.size() - 1), 0),
                   c2 = classSimilarity(p, tests, Math.min(optK, tests.size() - 1), 1);
            int resClass = (c1 > c2) ? 0 : 1;

            tm[resClass][p.geteClass()]++;
        }

        if (tm[1][1] == 0) {
            System.out.println("NAN");
            return new Pair<>(0., 0.);
        }
        double precision = 1.0 * tm[1][1] / (tm[1][1] + tm[1][0]),
               recall    = 1.0 * tm[1][1] / (tm[1][1] + tm[0][1]);

        return new Pair<>(2 * precision * recall / (precision + recall), fbMeasure(precision, recall));
    }

    private double fbMeasure(double precision, double recall) {
        double b = 0.5;
        if (Double.compare(precision, recall) < 0) {
            b = 2.0;
        }
        return (1 + b * b) * precision * recall / (b * b *precision + recall);
    }

    public Pair<Double, Double> learn() {
        List<Precedent> precedents = getPrecedents();
        if (precedents == null) {
            throw new DataException("Wrong data set file");
        } else if (precedents.stream().allMatch(Predicate.isEqual(precedents.get(0)))) {
            throw new DataException("Need all data equivalence classes different");
        }

        Collections.shuffle(precedents);

        // normalize if needed (ignore now)
        // need random shuffle divider here
        Divider<Precedent> divider = new CrossValidationDivider<>(precedents, KD_FOLD_CONST);
        divider.divide(0);
        List<Precedent> trains = divider.getTrainingSamples(),
                        tests = divider.getTestingSamples();

        return findMeasure(tests, findOptimalK(trains));
    }
}
