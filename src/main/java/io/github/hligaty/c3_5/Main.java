package io.github.hligaty.c3_5;


import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import io.github.hligaty.utils.StopWatch;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws TranslateException, IOException, InterruptedException {
        // 读取数据集
        int batchSize = 256;
        boolean randomShuffle = true;

        FashionMnist mnistTrain = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        FashionMnist mnistTest = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        mnistTrain.prepare();
        mnistTest.prepare();

        try (NDManager manager = NDManager.newBaseManager()) {
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            for (Batch batch : mnistTrain.getData(manager)) {
                NDArray x = batch.getData().head();
                NDArray y = batch.getLabels().head();
            }
            System.out.printf("write time:%.2f sec%n", stopWatch.stop());
        }
    }
}
