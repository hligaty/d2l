package io.github.hligaty.c4_2;

import ai.djl.Device;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import io.github.hligaty.utils.Training;

import java.io.IOException;

public class Main {
    /*
    初始化模型参数
     */
    static int batchSize = 256;
    
    static int numInputs = 784;
    static int numOutputs = 10;
    static int numHiddents = 256;
    
    static NDList params;
    
    public static void main(String[] args) throws TranslateException, IOException {
        
        /*
        数据集
         */
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
        
        try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "MXNet")) {
            /*
            初始化模型参数
             */
            NDArray W1 = manager.randomNormal(0, 0.01f, new Shape(numInputs, numHiddents), DataType.FLOAT32);
            NDArray b1 = manager.zeros(new Shape(numHiddents));
            NDArray W2 = manager.randomNormal(0, 0.01f, new Shape(numHiddents, numOutputs), DataType.FLOAT32);
            NDArray b2 = manager.zeros(new Shape(numOutputs));
            
            params = new NDList(W1, b1, W2, b2);

            for (NDArray param : params) {
                // 设置计算梯度
                param.setRequiresGradient(true);
            }

            /*
            损失函数
             */
            Loss loss = Loss.softmaxCrossEntropyLoss();
            
            /*
            训练
             */
            int numEpochs = Integer.getInteger("MAX_EPOCH", 10);
            float lr = 0.5f;

            double[] trainLoss = new double[numEpochs];
            double[] trainAccuracy = new double[numEpochs];
            double[] testAccuracy = new double[numEpochs];
            double[] epochCount = new double[numEpochs];

            float epochLoss = 0f;
            float accuracyVal = 0f;

            for (int epoch = 1; epoch <= numEpochs; epoch++) {
                System.out.print("Running epoch " + epoch + "...... ");
                // 迭代数据集
                for (Batch batch : trainIter.getData(manager)) {
                    NDArray X = batch.getData().head();
                    NDArray y = batch.getLabels().head();
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray yHat = net(X);

                        NDArray lossValue = loss.evaluate(new NDList(y), new NDList(yHat));
                        NDArray l = lossValue.mul(batchSize);

                        accuracyVal += Training.accuracy(yHat, y);
                        epochLoss += l.sum().getFloat();
                        
                        gc.backward(l);
                    }
                    batch.close();
                    Training.sgd(params, lr, batchSize);
                }

                trainLoss[epoch-1] = epochLoss/trainIter.size();
                trainAccuracy[epoch-1] = accuracyVal/trainIter.size();

                epochLoss = 0f;
                accuracyVal = 0f;
                // testing now
                for (Batch batch : testIter.getData(manager)) {

                    NDArray X = batch.getData().head();
                    NDArray y = batch.getLabels().head();

                    NDArray yHat = net(X); // net function call
                    accuracyVal += Training.accuracy(yHat, y);
                }

                testAccuracy[epoch-1] = accuracyVal/testIter.size();
                epochCount[epoch-1] = epoch;
                accuracyVal = 0f;
                System.out.println("Finished epoch " + epoch);
            }

            System.out.println("Finished training!");
        }
        
    }

    /*
    激活函数
     */
    static NDArray relu(NDArray X) {
        return X.maximum(0f);
    }

    /*
    模型
     */
    static NDArray net(NDArray X) {
        X = X.reshape(new Shape(-1, numInputs));
        NDArray H = relu(X.dot(params.get(0)).add(params.get(1)));
        return H.dot(params.get(2)).add(params.get(3));
    }
    
}
