package com.ifmo.ml.knn.dividers;

import java.util.List;

public class RandomCrossValidationDivider<T> extends CrossValidationDivider<T> {
    public RandomCrossValidationDivider(List<T> elements, int k) {
        super(elements, k);
    }
}
