package io.github.hligaty.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.training.dataset.ArrayDataset;

public class Util {
    // 读取数据集
    public static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle) {
        return new ArrayDataset.Builder()
                .setData(features)
                .optLabels(labels)
                .setSampling(batchSize, shuffle)
                .build();
    }
}
