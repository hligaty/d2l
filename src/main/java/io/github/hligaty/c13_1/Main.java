package io.github.hligaty.c13_1;

import ai.djl.basicdataset.cv.BananaDetection;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws TranslateException, IOException {
        BananaDetection.builder()
                .setSampling(1, true)
                .optUsage(Dataset.Usage.TRAIN).build().prepare();
    }
}
