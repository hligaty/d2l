package io.github.hligaty.c7_2;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.Sgd;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;

import java.io.IOException;

/**
 * VGG
 */
public class Main {
    public static void main(String[] args) throws TranslateException, IOException {
        int batchSize = 128;
        FashionMnist trainIter = FashionMnist.builder()
                .addTransform(new Resize(96))
                .addTransform(new ToTensor())
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, true)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();
        FashionMnist testIter = FashionMnist.builder()
                .addTransform(new Resize(96))
                .addTransform(new ToTensor())
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, true)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();
        trainIter.prepare();
        testIter.prepare();
        
        int[][] convArch = {{1, 64}, {1, 128}, {2, 256}, {2, 512}, {2, 512}};
        int ratio = 4;
        for (int i = 0; i < convArch.length; i++) {
            convArch[i][1] = convArch[i][1] / ratio;
        }

        try (Model model = Model.newInstance("vgg-tiny")) {
            SequentialBlock block = VGG(convArch);
            model.setBlock(block);

            SoftmaxCrossEntropyLoss loss = Loss.softmaxCrossEntropyLoss();

            Tracker lrt = Tracker.fixed(.05f);
            Sgd sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                    .optDevices(Engine.getInstance().getDevices(1))
                    .optOptimizer(sgd)
                    .addEvaluator(new Accuracy())
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            Trainer trainer = model.newTrainer(config);
            Shape inputShape = new Shape(1, 1, 96, 96);
            trainer.initialize(inputShape);

            EasyTrain.fit(trainer, 10, trainIter, testIter);
        }
    }

    static SequentialBlock vggBlock(int numConvs, int numChannels) {
        SequentialBlock block = new SequentialBlock();
        for (int i = 0; i < numConvs; i++) {
            block.add(Conv2d.builder()
                            .setFilters(numChannels)
                            .setKernelShape(new Shape(3, 3))
                            .optPadding(new Shape(1, 1))
                            .build())
                    .add(Activation::relu);
        }
        block.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        return block;
    }

    static SequentialBlock VGG(int[][] convArch) {
        SequentialBlock block = new SequentialBlock();
        for (int[] arch : convArch) {
            block.add(vggBlock(arch[0], arch[1]));
        }
        block
                .add(Blocks.batchFlattenBlock())
                .add(Linear.builder()
                        .setUnits(4096)
                        .build())
                .add(Activation::relu)
                .add(Dropout.builder()
                        .optRate(.5f)
                        .build())
                .add(Linear.builder()
                        .setUnits(4096)
                        .build())
                .add(Activation::relu)
                .add(Dropout.builder()
                        .optRate(.5f)
                        .build())
                .add(Linear.builder()
                        .setUnits(10)
                        .build());
        return block;
    }
}
