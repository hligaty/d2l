package io.github.hligaty.c4_6;

import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.translate.TranslateException;
import io.github.hligaty.utils.Training;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;
import tech.tablesaw.plotly.components.Figure;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

public class Main {
    static NDManager manager;
    static int numInputs = 784;
    static int numOutputs = 10;
    static int numHiddens1 = 256;
    static int numHiddens2 = 256;
    static int numHiddens3 = 128;
    static NDArray W1;
    static NDArray b1;
    static NDArray W2;
    static NDArray b2;
    static NDArray W3;
    static NDArray b3;
    static NDArray W4;
    static NDArray b4;
    static NDList params;
    
    
    static {
        manager = NDManager.newBaseManager();
        W1 = manager.randomNormal(0, 0.01f, new Shape(numInputs, numHiddens1), DataType.FLOAT32);
        b1 = manager.zeros(new Shape(numHiddens1));
        W2 = manager.randomNormal(0, 0.01f, new Shape(numHiddens1, numHiddens2), DataType.FLOAT32);
        b2 = manager.zeros(new Shape(numHiddens2));
        W3 = manager.randomNormal(0, 0.01f, new Shape(numHiddens2, numHiddens3), DataType.FLOAT32);
        b3 = manager.zeros(new Shape(numHiddens3));
        W4 = manager.randomNormal(0, 0.01f, new Shape(numHiddens3, numOutputs), DataType.FLOAT32);
        b4 = manager.zeros(new Shape(numOutputs));
        params = new NDList(W1, b1, W2, b2, W3, b3, W4, b4);
    }

    public static void main(String[] args) throws TranslateException, IOException {
        for (NDArray param : params) {
            param.setRequiresGradient(true);
        }

        int numEpochs = Integer.getInteger("MAX_EPOCH", 10);
        float lr = 0.5f;
        int batchSize = 256;

        double[] trainLoss = new double[numEpochs];
        double[] trainAccuracy = new double[numEpochs];
        double[] testAccuracy = new double[numEpochs];
        double[] epochCount = new double[numEpochs];

        Loss loss = new SoftmaxCrossEntropyLoss();

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

        float epochLoss = 0f;
        float accuracyVal = 0f;

        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            // Iterate over dataset
            System.out.print("Running epoch " + epoch + "...... ");
            for (Batch batch : trainIter.getData(manager)) {
                NDArray X = batch.getData().head();
                NDArray y = batch.getLabels().head();

                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                    NDArray yHat = net(X, true); // net function call

                    NDArray lossValue = loss.evaluate(new NDList(y), new NDList(yHat));
                    NDArray l = lossValue.mul(batchSize);

                    epochLoss += l.sum().getFloat();

                    accuracyVal += Training.accuracy(yHat, y);
                    gc.backward(l); // gradient calculation
                }

                batch.close();
                Training.sgd(params, lr, batchSize); // updater
            }

            trainLoss[epoch - 1] = epochLoss / trainIter.size();
            trainAccuracy[epoch - 1] = accuracyVal / trainIter.size();

            epochLoss = 0f;
            accuracyVal = 0f;

            for (Batch batch : testIter.getData(manager)) {
                NDArray X = batch.getData().head();
                NDArray y = batch.getLabels().head();

                NDArray yHat = net(X, false); // net function call
                accuracyVal += Training.accuracy(yHat, y);
            }

            testAccuracy[epoch - 1] = accuracyVal / testIter.size();
            epochCount[epoch - 1] = epoch;
            accuracyVal = 0f;
            System.out.println("Finished epoch " + epoch);
        }

        System.out.println("Finished training!");

        String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

        Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
        Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
        Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length,
                trainLoss.length + testAccuracy.length + trainAccuracy.length, "test acc");

        Table data = Table.create("Data").addColumns(
                DoubleColumn.create("epochCount", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))),
                DoubleColumn.create("loss", ArrayUtils.addAll(trainLoss, ArrayUtils.addAll(trainAccuracy, testAccuracy))),
                StringColumn.create("lossLabel", lossLabel)
        );

        Figure figure = LinePlot.create("", data, "epochCount", "loss", "lossLabel");
        Plot.show(figure, Paths.get("3epoch-0.2&0.5.html").toFile());
    }

    static float dropout1 = 0.2f;
    static float dropout2 = 0.5f;
    static float dropout3 = 0.7f;

    static NDArray dropoutLayer(NDArray X, float dropout) {
        // 为 1 时全部删除
        if (dropout == 1.0f) {
            return manager.zeros(X.getShape());
        }
        // 为 0 时全部保留
        if (dropout == 0f) {
            return X;
        }

        NDArray mask = manager.randomUniform(0f, 1.0f, X.getShape()).gt(dropout);
        return mask.toType(DataType.FLOAT32, false).mul(X).div(1.0f - dropout);
    }

    static NDArray net(NDArray X, boolean isTraining) {

        X = X.reshape(-1, numInputs);
        NDArray H1 = Activation.relu(X.dot(W1).add(b1));

        if (isTraining) {
            H1 = dropoutLayer(H1, dropout1);
        }

        NDArray H2 = Activation.relu(H1.dot(W2).add(b2));
        if (isTraining) {
            H2 = dropoutLayer(H2, dropout2);
        }

        NDArray H3 = Activation.relu(H2.dot(W3).add(b3));
        if (isTraining) {
            H3 = dropoutLayer(H3, dropout2);
        }
        
        return H3.dot(W4).add(b4);
    }
}
