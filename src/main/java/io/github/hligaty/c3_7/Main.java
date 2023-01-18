package io.github.hligaty.c3_7;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
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
    static int batchSize = 256;
    static boolean randomShuffle = true;
    
    /*
    实现 softmax 回归比较复杂，深度学习框架在一些著名的技巧外采取了额外的预防措施，
    来保证数值的稳定性，避免从零写模型时可能遇到的陷阱
     */
    public static void main(String[] args) throws TranslateException, IOException {
        FashionMnist trainingSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();


        FashionMnist validationSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, false)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "MXNet");
             Model model = Model.newInstance("soft-regression")) {
            SequentialBlock net = new SequentialBlock();
            net.add(Blocks.batchFlattenBlock(28 * 28));
            net.add(Linear.builder().setUnits(10).build());
            
            model.setBlock(net);

            // 在交叉熵损失函数中传递未归一化的预测，并同时计算softmax及其对数
            Loss loss = Loss.softmaxCrossEntropyLoss();

            // 优化算法
            Tracker lrt = Tracker.fixed(0.1f);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();
            
            // Trainer 初始化配置
            DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                    .optOptimizer(sgd)
                    .optDevices(manager.getEngine().getDevices())
                    .addEvaluator(new Accuracy())
                    .addTrainingListeners(TrainingListener.Defaults.logging());

            Trainer trainer = model.newTrainer(config);

            // 初始化模型参数
            trainer.initialize(new Shape(1, 28 * 28));
            
            // 运行性能指标
            Metrics metrics = new Metrics();
            trainer.setMetrics(metrics);
            
            // 训练
            int numEpochs = 3;
            EasyTrain.fit(trainer, numEpochs, trainingSet, validationSet);
        }
    }
    
    static class ActivationFunction {
        public static NDList softmax(NDList arrays) {
            return new NDList(arrays.singletonOrThrow().logSoftmax(1));
        }
    }
}
