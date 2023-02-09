package io.github.hligaty.c6_6;


import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.IOException;

public class Main {
    public static void main(String[] args) throws TranslateException, IOException {
        int batchSize = 256;

        FashionMnist trainIter = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, true)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();
        FashionMnist testIter = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, true)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();
        trainIter.prepare();
        testIter.prepare();

        SequentialBlock block = new SequentialBlock();

        block.add(Conv2d.builder()
                        .setKernelShape(new Shape(5, 5))
                        .optPadding(new Shape(2, 2))
                        .optBias(false)
                        .setFilters(6)
                        .build())
                .add(Activation::sigmoid)
                .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                .add(Conv2d.builder()
                        .setKernelShape(new Shape(5, 5))
                        .setFilters(16)
                        .build())
                .add(Activation::sigmoid)
                .add(Pool.avgPool2dBlock(new Shape(5, 5), new Shape(2, 2), new Shape(2, 2)))
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder()
                        .setUnits(120)
                        .build())
                .add(Linear.builder()
                        .setUnits(84)
                        .build())
                .add(Linear.builder()
                        .setUnits(10)
                        .build());
        
        float lr = .9f;
        try (Model model = Model.newInstance("LeNet")) {
            model.setBlock(block);

            Loss loss = Loss.softmaxCrossEntropyLoss();

            Tracker lrt = Tracker.fixed(lr);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                    .optOptimizer(sgd)
                    .optDevices(Engine.getInstance().getDevices(1))
                    .addEvaluator(new Accuracy())
                    .addTrainingListeners(TrainingListener.Defaults.basic());

            Trainer trainer = model.newTrainer(config);

            //Training.trainingChapter6(trainIter, testIter, numEpochs, trainer, )
        }
    }
}
