package io.github.hligaty.c3_2;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class DataPoints {
    private NDArray x, y;

    public DataPoints(NDArray x, NDArray y) {
        this.x = x;
        this.y = y;
    }

    public NDArray getX() {
        return x;
    }

    public NDArray getY() {
        return y;
    }

    // Generate y = X w + b + noise
    public static DataPoints syntheticData(NDManager manager, NDArray w, float b, int numExamples) {
        NDArray X = manager.randomNormal(new Shape(numExamples, w.size()));
        NDArray y = X.dot(w).add(b);
        y.addi(manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32));
        return new DataPoints(X, y);
    }
}
