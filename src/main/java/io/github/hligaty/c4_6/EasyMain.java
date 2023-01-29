package io.github.hligaty.c4_6;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.metric.Metric;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.Sgd;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class EasyMain {
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
        
        /*
        模型，两个隐藏层，各两个 relu 激活函数和 dropout 正则项
         */
        SequentialBlock net = new SequentialBlock();
        net.add(Blocks.batchFlattenBlock(784));
        net.add(Linear.builder().setUnits(256).build());
        net.add(Activation::relu);
        net.add(Dropout.builder().optRate(0.5f).build());
        net.add(Linear.builder().setUnits(256).build());
        net.add(Activation::relu);
        net.add(Dropout.builder().optRate(0.2f).build());
        net.add(Linear.builder().setUnits(10).build());
        /*
        学习率
         */
        Tracker lrt = Tracker.fixed(0.5f);
        /*
        优化函数
         */
        Sgd sgd = Optimizer.sgd().optWeightDecays(0.001f).setLearningRateTracker(lrt).build();
        /*
        损失函数
         */
        SoftmaxCrossEntropyLoss loss = Loss.softmaxCrossEntropyLoss();
        /*
        配置训练过程
         */
        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optDevices(Engine.getInstance().getDevices(1))
                .optInitializer(new NormalInitializer(0.01f), Parameter.Type.WEIGHT)
                .optOptimizer(sgd)
                .addEvaluator(new Accuracy())
                .addTrainingListeners(TrainingListener.Defaults.logging());

        Map<String, double[]> evaluatorMetrics = new HashMap<>();
        
        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(net);
            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, 784));
                trainer.setMetrics(new Metrics());

                int numEpochs = 10;
                EasyTrain.fit(trainer, numEpochs, trainIter, testIter);


                Metrics metrics = trainer.getMetrics();
                trainer.getEvaluators()
                        .forEach(evaluator -> {
                            evaluatorMetrics.put("train_epoch_" + evaluator.getName(), metrics.getMetric("train_epoch_" + evaluator.getName()).stream()
                                    .mapToDouble(Metric::getValue).toArray());
                            evaluatorMetrics.put("validate_epoch_" + evaluator.getName(), metrics.getMetric("validate_epoch_" + evaluator.getName()).stream()
                                    .mapToDouble(Metric::getValue).toArray());
                        });


                double[] epochCount = trainer.getManager().arange(1, numEpochs + 1, 1, DataType.FLOAT64).toDoubleArray();

                double[] trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
                double[] trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
                double[] testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

                String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

                Arrays.fill(lossLabel, 0, trainLoss.length, "test acc");
                Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
                Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length,
                        trainLoss.length + testAccuracy.length + trainAccuracy.length, "train loss");

                Table data = Table.create("Data").addColumns(
                        DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))),
                        DoubleColumn.create("loss", ArrayUtils.addAll(testAccuracy , ArrayUtils.addAll(trainAccuracy, trainLoss))),
                        StringColumn.create("lossLabel", lossLabel)
                );

                Plot.show(LinePlot.create("", data, "epochCount", "loss", "lossLabel"),
                        Paths.get("5-2.html").toFile());
            }
        }
    }
}
